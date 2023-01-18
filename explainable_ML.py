import shap
import interpret.glassbox

class explaining_ML:
    def __init__(self, sample_ind):
        self.sample_ind = sample_ind

    def MLexplainer(self, model, xtrain):
        explainer = shap.Explainer(model.predict, xtrain)
        shap_values = explainer(xtrain)
        return explainer, shap_values

    def partial_dependency_plotting(self, model, xtrain, shap_values, variable):
        # make a standard partial dependence plot
        shap.partial_dependence_plot(
            variable, model.predict, xtrain, model_expected_value=True,
            feature_expected_value=True, ice=False,
            shap_values=shap_values[self.sample_ind:self.sample_ind+1,:]
        )
        plt.show()

    def plotting_waterfall(self, shap_values):
           shap.plots.waterfall(shap_values[self.sample_ind], show = True)
        plt.show()

        
    def model_interpret(self, xtrain, ytrain, shap_values):
        model_ebm = interpret.glassbox.ExplainableBoostingRegressor(interactions=0)
        model_ebm.fit(xtrain, ytrain)
        # explain the GAM model with SHAP
        explainer_ebm = shap.Explainer(model_ebm.predict, xtrain)
        shap_values_ebm = explainer_ebm(xtrain)
        # the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
        shap.plots.beeswarm(shap_values_ebm)
        shap.plots.beeswarm(shap_values)   
        shap.plots.bar(shap_values)
        plt.show()

 
    def summarized_plot(self, shap_values):
        shap.summary_plot(shap_values)
        plt.show()

