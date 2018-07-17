#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

import inspect
import os
import tensorflow as tf
from tf_container.run import logger
import tf_container.s3_fs as s3_fs
from bayes_opt.bayesian_optimization import BayesianOptimization


class Trainer(object):
    DEFAULT_TRAINING_CHANNEL = 'training'

    def __init__(self,
                 customer_script,
                 current_host,
                 hosts,
                 train_steps=1000,
                 eval_steps=100,
                 input_channels=None,
                 model_path=None,
                 output_path=None,
                 customer_params={},
                 save_checkpoints_secs=300):
        """

        Args:
            customer_script: (module) Customer loaded module
            current_host: (str) Current hostname
            hosts: list (str) List with all containers list names
            train_steps: (int) Perform this many steps of training. 'None', the default,
                means train forever.
            eval_steps: (int) 'evaluate' runs until input is exhausted (or another exception
                is raised), or for 'eval_steps' steps, if specified.
            input_channels: (dict) Dictionary with input channels
            model_path: (str) Directory where checkpoints will be saved. Can be a S3 bucket
            output_path: (str) Local directory where the model will be saved
        """
        self.customer_script = customer_script
        self.current_host = current_host
        self.hosts = hosts
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.input_channels = input_channels
        self.model_path_base = model_path
        self.ouput_path = output_path
        self.task_type = None

        customer_params['save_checkpoints_secs'] = customer_params.get('save_checkpoints_secs', save_checkpoints_secs)

        self.customer_params = customer_params

        if model_path.startswith('s3://'):
            s3_fs.configure_s3_fs(model_path)
    
        
    def params2Path(self, params):
        return "-".join(["%s_%.2f" % (str(paramName), float(paramVal)) for paramName, paramVal in sorted(params.items(), key=lambda x:x[0])])

    def train(self):
        
        if "tunning" not in self.customer_params:
            self.model_path = self.model_path_base
            self.trainModelOnParams()
            return 
        exploratoryParams =  self.customer_params['tunning']
        def addParamsTrain(**params):
            self.customer_params.update(params)
            self.model_path = os.path.join(self.model_path_base, self.params2Path(params))
            estimator = self.trainModelOnParams()
            invoke_args = self._resolve_input_fn_args(self.customer_script.eval_input_fn)
            res = estimator.evaluate(lambda: self.customer_script.eval_input_fn(**invoke_args))
            return res['accuracy']
            
        nnBO = BayesianOptimization(addParamsTrain, exploratoryParams)
        ## do a first exploratory work with many init and big kappa
        exploratory_tunning = self.customer_params['exploratory_tunning'] if 'exploratory_tunning' in self.customer_params else 6
        nnBO.maximize(init_points=2, n_iter=exploratory_tunning, kappa=5, acq='ei')
        #finetune with small kappa
        fine_tunning = self.customer_params['fine_tunning'] if 'fine_tunning' in self.customer_params else 6
        nnBO.maximize(init_points=0, n_iter=fine_tunning, kappa=2, acq='ei')
        
        logger.info( "-------------------")
        logger.info( "model results")
        for params, results in zip(nnBO.res['all']['params'], nnBO.res['all']['values']):
            logger.info( "%s : %f" % (str(params), results))
        logger.info( "-------------------")
        logger.info( "best model")
        logger.info(nnBO.res['max']['max_params'])
        logger.info(nnBO.res['max']['max_val'])
        logger.info( "-------------------")
        best_model_path = os.path.join(self.model_path_base, self.params2Path(nnBO.res['max']['max_params']))
        logger.info("best_model_path=%s" % best_model_path)
        return best_model_path
        
    def trainModelOnParams(self):
        run_config = self._build_run_config()
        train_spec = self._build_train_spec()
        eval_spec = self._build_eval_spec()
        
        estimator = self._build_estimator(run_config=run_config)
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
        return estimator
        

    def _build_run_config(self):
        valid_runconfig_keys = ['save_summary_steps', 'save_checkpoints_secs', 'save_checkpoints_steps',
                                'keep_checkpoint_max', 'keep_checkpoint_every_n_hours', 'log_step_count_steps']

        runconfig_params = {k: v for k, v in self.customer_params.items() if k in valid_runconfig_keys}

        logger.info('creating RunConfig:')
        logger.info(runconfig_params)

        run_config = tf.estimator.RunConfig(model_dir=self.model_path, **runconfig_params)
        return run_config

    def _build_estimator(self, run_config):
        hyperparameters = self.customer_params

        if hasattr(self.customer_script, 'estimator_fn'):
            logger.info('invoking the user-provided estimator_fn')
            return self.customer_script.estimator_fn(run_config, hyperparameters)
        elif hasattr(self.customer_script, 'keras_model_fn'):
            logger.info('invoking the user-provided keras_model_fn')
            model = self.customer_script.keras_model_fn(hyperparameters)
            return tf.keras.estimator.model_to_estimator(keras_model=model, config=run_config)
        else:
            logger.info('creating an estimator from the user-provided model_fn')

            # We must wrap the model_fn from customer_script like this to maintain compatibility with our
            # existing behavior, which passes arguments to the customer model_fn positionally, not by name.
            # The TensorFlow Estimator checks the signature of the given model_fn for a parameter named "params":
            # https://github.com/tensorflow/tensorflow/blob/2c9a67ffb384a13cd533a0e89a96211058fa2631/tensorflow/python/estimator/estimator.py#L1215
            # Wrapping it in _model_fn allows the customer to use whatever parameter names they want. It's unclear whether
            # this behavior is desirable theoretically, but we shouldn't break existing behavior.

            def _model_fn(features, labels, mode, params):
                return self.customer_script.model_fn(features, labels, mode, params)

            return tf.estimator.Estimator(
                model_fn=_model_fn,
                params=hyperparameters,
                config=run_config)

    def _build_train_spec(self):
        invoke_args = self._resolve_input_fn_args(self.customer_script.train_input_fn)
        train_input_fn = lambda: self.customer_script.train_input_fn(**invoke_args)

        return tf.estimator.TrainSpec(train_input_fn, max_steps=self.train_steps)

    def saves_training(self):
        return hasattr(self.customer_script, 'serving_input_fn')

    def _build_eval_spec(self):
        invoke_args = self._resolve_input_fn_args(self.customer_script.eval_input_fn)
        eval_input_fn = lambda: self.customer_script.eval_input_fn(**invoke_args)

        if self.saves_training():
            serving_input_receiver_fn = lambda: self.customer_script.serving_input_fn(self.customer_params)
            exporter = tf.estimator.LatestExporter('Servo',
                                                   serving_input_receiver_fn=serving_input_receiver_fn)
        else:
            logger.warn('serving_input_fn not specified, model NOT saved, use checkpoints to reconstruct')
            exporter = None

        valid_eval_keys = ['start_delay_secs', 'throttle_secs']
        eval_params = {k: v for k, v in self.customer_params.items() if k in valid_eval_keys}

        return tf.estimator.EvalSpec(eval_input_fn, steps=self.eval_steps, exporters=exporter, **eval_params)

    def _resolve_input_fn_args(self, customer_fn):
        declared_args = inspect.getargspec(customer_fn)
        return {arg: self._resolve_input_fn_param_value(arg) for arg in declared_args.args}

    def _resolve_input_fn_param_value(self, alias_key):
        """
        Handle potentially aliased key name and return value for valid one.

        :return: value for the requested parameter or None
        """
        key_mappings = {('training_dir', 'dir'): 'training_dir',
                        ('hyperparameters', 'params'): 'hyperparameters',
                        ('input_channels', 'channels'): 'input_channels'}
        resolved_key = None
        for k, v in key_mappings.items():
            if alias_key in k:
                resolved_key = v
                break

        parameter_values = {'training_dir': self.input_channels.get(self.DEFAULT_TRAINING_CHANNEL, None),
                            'hyperparameters': self.customer_params,
                            'input_channels': self.input_channels}
        return parameter_values[resolved_key] if resolved_key else None

    def build_tf_config(self):
        """Builds a dictionary containing cluster information based on number of hosts and number of parameter servers.
        More information about TF_Config: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn
        /python/learn/estimators/run_config.py#L77
        :return: task_type and tf_config dictionary
        """

        masters = self.hosts[:1]
        workers = self.hosts[1:]
        ps = self.hosts[:] if len(self.hosts) > 1 else None

        self.task_type = self._get_task_type(masters)

        task_map = {'master': masters, 'worker': workers}

        if ps:
            task_map['ps'] = ps

        task_id = task_map[self.task_type].index(self.current_host)

        def build_host_addresses(my_hosts, port='2222'):
            return ['{}:{}'.format(host, port) for host in my_hosts]

        tf_config = {
            'cluster': {
                'master': build_host_addresses(masters),
            },
            'task': {
                'index': task_id,
                'type': self.task_type
            },
            'environment': 'cloud'
        }

        if ps:
            tf_config['cluster']['ps'] = build_host_addresses(ps, port='2223')

        if len(workers) > 0:
            tf_config['cluster']['worker'] = build_host_addresses(workers)

        return tf_config

    def _get_task_type(self, masters):
        if self.current_host in masters:
            return 'master'
        return 'worker'
