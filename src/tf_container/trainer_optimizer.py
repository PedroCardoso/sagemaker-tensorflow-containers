
import os
from tf_container.run import logger
from bayes_opt.bayesian_optimization import BayesianOptimization
from tf_container.trainer import Trainer


class TrainerBayesOptimizer(Trainer):
    DEFAULT_TRAINING_CHANNEL = 'training'
        
    def params2Path(self, params):
        return "-".join(["%s_%.2f" % (str(paramName), float(paramVal)) for paramName, paramVal in sorted(params.items(), key=lambda x:x[0])])

    def train(self):
        
        if "tunning" not in self.customer_params:
            super(TrainerBayesOptimizer, self).train()
            return self.model_path
        exploratoryParams =  self.customer_params['tunning']
        self.model_path_base = self.model_path
        def addParamsTrain(**params):
            self.customer_params.update(params)
            self.model_path = os.path.join(self.model_path_base, self.params2Path(params))
            estimator = super(TrainerBayesOptimizer, self).train()
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
        

