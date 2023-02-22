import pandas as pd
import numpy as np
import os
import copy
import joblib
from io import StringIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.calibration import *
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import plot_partial_dependence
import shap
from pdpbox import pdp, info_plots
import statsmodels.api as sm
from lime.lime_tabular import LimeTabularExplainer


# ---------------------------------------------------------------------------------------
# Explainer Class:
#
# Global & Local
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class Explainer:

    _model = None

    def __init__(self, r_model):
        """
        Explainer Constructor
        """
        self._model = r_model
        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_output = self.p_project + r'\output\output_parser'
        self.p_classifier = self.p_project + r'\output\output_classifier'
        self.p_models = self.p_classifier + r'\models'
        if self._model.b_vpn:
            self.p_project = self._model.set_vpn_dir(self.p_project)
            self.p_output = self._model.set_vpn_dir(self.p_output)
            self.p_classifier = self._model.set_vpn_dir(self.p_classifier)
            self.p_models = self._model.set_vpn_dir(self.p_models)
        self.l_targets_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']

    def load_model(self, curr_label):
        """
        function loads model from disk
        :param curr_label model name
        """
        p_load = self._model.validate_path(self.p_models, curr_label, 'pkl')
        o_model = joblib.load(p_load)
        filename = self._model.get_filename(p_load)
        print(f'Model {filename} has been loaded.')
        return o_model

    def load_data(self, curr_label):
        """
        function loads training and test sets from disk
        :param curr_label set name
        """
        p_x_train = self._model.validate_path(self.p_models, curr_label+'_x_train', 'csv')
        p_x_test = self._model.validate_path(self.p_models, curr_label+'_x_test', 'csv')
        p_y_train = self._model.validate_path(self.p_models, curr_label+'_y_train', 'csv')
        p_y_test = self._model.validate_path(self.p_models, curr_label+'_y_test', 'csv')
        x_train = pd.read_csv(p_x_train)
        # x_test = pd.read_csv(p_x_test)
        x_test = None
        # y_train = pd.read_csv(p_y_train)
        y_train = None
        # y_test = pd.read_csv(p_y_test)
        y_test = None
        return x_train, x_test, y_train, y_test

    def rf_regressor(self, x_train, y_train, m_rf):
        """
            (1) Global Interpretation
            (1.1) Feature Importance
            (1.1.1) Random Forest Regressor
        """
        i_top = 10
        feature_importance = pd.DataFrame(columns=['Variable', 'Importance'])
        feature_importance['Variable'] = x_train.columns
        feature_importance['Importance'] = m_rf.feature_importances_
        feature_importance.sort_values(by='Importance', ascending=False).head(i_top)

    def lasso_selection(self, x_train, x_test, y_train, y_test):
        """
            (1.1.2) Lasso
            finds optimal alpha
        """
        x_train = x_train.copy()
        x_train = x_train.astype('float64')
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)

        search = GridSearchCV(Lasso(),
                              {'alpha': np.arange(0.1, 200, 1)},
                              cv=5, scoring='neg_mean_squared_error', verbose=0
                              )
    
        search.fit(x_train, y_train)
        search.best_params_
        f_alpha = search.best_params_['alpha']
        print(f'Alpha value found: {f_alpha}')
    
        # build model
        lasso_reg = Lasso(alpha=f_alpha)  # weight penalty (if too high: w->0)
        lasso_reg.fit(x_train, y_train)
        lasso_pred = lasso_reg.predict(x_test)
    
        # scores
        MSE = round(mean_squared_error(lasso_pred, y_test))
        MAE = round(mean_absolute_error(lasso_pred, y_test))
        r2 = round(r2_score(lasso_pred, y_test), 2)
        print(f'MSE: {MSE} | MAE: {MAE} | R2score: {r2}')
    
        # plot
        coefficients = lasso_reg.coef_
        importance = np.abs(coefficients)
        plt.figure(figsize=(14, 6))
        plt.bar(x_train.columns, importance)
        plt.title("Feature Importance (Lasso)", fontsize=25)
        plt.ylabel("t-statistic (absolute value)", fontsize=18)
        plt.grid()
        plt.xticks(rotation=90, fontsize=15)
        plt.show()

    def gsm(self, x_train, m_rf):
        """
            (1.2) Global Surrogate Model
            (1.2.1) GSM by Decision Tree Regressor
        """
        y_preds_train = m_rf.predict(x_train)  # saves preds of training set
        m_dt = DecisionTreeRegressor(max_depth=4, random_state=10)  # interpretable decision tree model
        m_dt.fit(x_train, y_preds_train)  # fitting the surrogate decision tree model using the training set and new target
        decision_tree = tree.export_graphviz(m_dt, out_file='tree.dot', feature_names=x_train.columns, filled=True,
                                             max_depth=4)  # visualization
        # !dot - Tpng
        # tree.dot - o
        # tree.png  # converts the dot image to png format

        image = plt.imread('tree.png')  # plot
        plt.figure(figsize=(25, 25))
        plt.imshow(image)

        decision_tree = tree.export_graphviz(m_dt, out_file='tree.dot', feature_names=x_train.columns, filled=True,
                                             max_depth=2)  # visualization
        # !dot - Tpng
        # tree.dot - o
        # tree.png  # converts the dot image to png format

        image = plt.imread('tree.png')  # plot
        plt.figure(figsize=(25, 25))
        plt.imshow(image)

    def shap_global(self, x_train, m_xgb):
        """
            (1.3) SHAP
            (1.3.1) SHAP Summary/Beeswarm Plot
        """
        shap_values = shap.TreeExplainer(m_xgb).shap_values(x_train)
        shap.summary_plot(shap_values, x_train, feature_names=x_train.columns)

        # SHAP Bar Plot
        shap.summary_plot(shap_values, x_train, plot_type="bar", feature_names=x_train.columns)

    def pdp_single(self, x_test, m_rf):
        """
            (1.4) Partial Dependency Plot
            (1.4.1) Single Feature
        """

        # (1.4.1.1) PDP - Single
        s_feature = 'age'
        pdp_single = pdp.pdp_isolate(model=m_rf, dataset=x_test, model_features=x_test.columns, feature=s_feature)

        # plot for numerical variables
        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_single, feature_name=s_feature + '_mean')

        # plot for categorical variables
        # fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_single,
        #                          feature_name=s_feature+'_mean',
        #                          center=True,
        #                          plot_lines=True,
        #                          frac_to_plot=100,
        #                          plot_pts_dist=True)

        plt.show()

        pdp_single = plot_partial_dependence(m_rf,
                                             features=[s_feature],
                                             X=x_test,
                                             grid_resolution=10)  # number of values to plot on x axis

    def pdp_double(self, x_train, x_test, m_xgb):
        """
            (1.4.2) Feature Pair
            (1.4) PDP - Pair
        """
        l_pair = ['age', 'gender']
        inter = pdp.pdp_interact(model=m_xgb, dataset=x_test, model_features=x_train.columns, features=l_pair)
        pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=l_pair, plot_type='contour', x_quantile=True,
                              plot_pdp=True)
        plt.show()

    def weight_plot(self, x_train, y_train):
        """
            (1.4.3) Weight Plot: by linear regression coefficient values and standard errors
        """
        m_lr = sm.OLS(y_train, x_train)  # first argument is y
        results = m_lr.fit()  # results.params
        error = results.params - results.conf_int()[0]
        coef_df = pd.DataFrame({'coefficient': results.params.values[1:],  # drops the intercept
                                'error': error.values[1:],
                                'variable': error.index.values[1:]
                                })

        # coef_df = pd.DataFrame({'coefficient': round(results.params),
        #                         'standard wrror': round(results.bse),
        #                         't_stats': round(results.tvalues, 1),
        #                         'error': round(error)
        #                        }).reset_index().rename(columns={'index': 'variable'})

        # plot
        coef_df.plot(y='coefficient', x='variable', kind='bar', color='none', yerr='error', legend=False, figsize=(12, 8))
        plt.scatter(x=np.arange(coef_df.shape[0]), s=100, y=coef_df['coefficient'], color='blue')
        plt.axhline(y=0, linestyle='--', color='black', linewidth=1)
        plt.title('Weight Plot: Coefficient and Standard Error')
        plt.show()

    def local_lime(self, x_train, x_test, y_train, m_rf):
        """
            (2) Local Interpretation
            (2.1) Lime
        """
        # i_record = np.random.randint(0, np.array(df_X_test_public).shape[0])
        i_record = 5  # use 1 arbitrary row of data from X_test

        explainer = LimeTabularExplainer(x_train.values, mode="regression", feature_names=x_train.columns)
        X_observation = x_test.iloc[[i_record], :]
        m_rf.predict(X_observation)[0]
        explanation = explainer.explain_instance(X_observation.values[0], m_rf.predict)

        explanation.show_in_notebook(show_table=True, show_all=False)
        print(f'Explanaition Score: {explanation.score:.2f}')

    def local_shap(self, x_train, x_test, y_train, m_rf):
        """
            (2.2) SHAP: By Record
            (2.2) SHAP Force Plot
        """
        explainer = shap.TreeExplainer(m_rf)
        shap.initjs()
        i_record = 5  # use 1 arbitrary row of data from X_test
        shap_values = explainer.shap_values(x_test)
        shap.force_plot(explainer.expected_value, shap_values[i_record, :], x_test.iloc[0, :], feature_names=x_test.columns)
        # shap.force_plot(explainer.expected_value, shap_values[0,:100], x_train.iloc[0,:100])

    def run(self):
        l_targets = ['A+B']
        for curr_label in l_targets:
            s_extension = ''
            s_extension = '_diff_0'
            o_model = self.load_model(curr_label+s_extension)
            x_train, x_test, y_train, y_test = self.load_data(curr_label+s_extension)

            m_rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=-1, random_state=10)
            m_rf.fit(x_train, y_train)

            m_xgb = XGBClassifier(max_depth=4, eta=0.02, n_estimators=20, random_state=42, objective='binary:logistic')
            m_xgb.fit(x_train, y_train)

            # global #
            self.rf_regressor(x_train, y_train, m_rf)
            self.lasso_selection(x_train, x_test, y_train, y_test)
            self.gsm(x_train, m_rf)
            self.shap_global(x_train, o_model)
            self.pdp_single(x_test, m_rf)
            self.pdp_double(x_train, x_test, m_xgb)
            self.weight_plot(x_train, y_train)

            # local #
            self.local_lime(x_train, x_test, y_train, m_rf)
            self.local_shap(x_train, x_test, y_train, m_rf)
