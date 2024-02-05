import logging
from time import time
from typing import Any, Dict, Tuple
from datasieve.pipeline import Pipeline
import numpy as np
import numpy.typing as npt
from pandas import DataFrame
import pandas as pd
from h2o import H2OFrame
import h2o
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, r2_score
import datasieve.transforms as ds
from datasieve.transforms import SKLearnWrapper
from sklearn.feature_selection import SelectKBest
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)
h2o.init()


class FreqaiH2OAutoML(IFreqaiModel):
    """
    User created prediction model. The class inherits IFreqaiModel, which
    means it has full access to all Frequency AI functionality. Typically,
    users would use this to override the common `fit()`, `train()`, or
    `predict()` methods to add their custom data handling tools or change
    various aspects of the training that cannot be configured via the
    top level config.json file.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary holding all data for train, test,
            labels, weights
        :param dk: The datakitchen object for the current coin/model
        """

        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0].ravel()

        train_features_df = pd.DataFrame(data_dictionary["train_features"])
        train_labels_df = pd.DataFrame(data_dictionary["train_labels"])
        train_weights_df = pd.DataFrame(data_dictionary["train_weights"])

        nfolds = self.freqai_info.get('nfolds', 5)
        fold_size = len(train_labels_df) // nfolds
        fold_col = np.repeat(np.arange(nfolds), fold_size)
        train_labels_df['fold_numbers'] = np.append(fold_col, 
                                                    np.arange(len(train_labels_df) % nfolds))
        
        training_frame = H2OFrame(pd.concat([train_features_df, train_labels_df, train_weights_df], axis=1))

        # Concatenate DataFrames for validation/blending
        validation_features_df = pd.DataFrame(data_dictionary["test_features"])
        validation_labels_df = pd.DataFrame(data_dictionary["test_labels"])
        validation_frame = H2OFrame(pd.concat([validation_features_df, validation_labels_df], axis=1))
        
        model = h2o.automl.H2OAutoML(**self.model_training_parameters)            
        
        model.train(
            x = train_features_df.columns.tolist(), 
            y = train_labels_df.columns.tolist()[0],
            training_frame = training_frame,
            blending_frame = validation_frame, #used for Stacked Ensemble training in binary classification and regression
            fold_column="fold_numbers"
        )
        
        if self.freqai_info.get("continual_learning", False):
            logger.warning("Continual learning is not supported for "
                           "H2OAutoML, ignoring.")
        
        if hasattr(self, "class_names"): # only define for classifiers since AutoML can do classification or regression tasks
            model._classes = self.class_names 
        
        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) == 0:
            eval_set = None

        elif self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0 and hasattr(self, "class_names"): #classification report
            test_features = H2OFrame(pd.DataFrame(data_dictionary["test_features"]))
            preds=model.leader.predict(test_features).as_data_frame()
            eval_set = (pd.DataFrame(data_dictionary["test_features"]),
                        pd.DataFrame(data_dictionary["test_labels"].to_numpy()[:, 0]))
    
            logger.info("-------------------- Classification Report --------------------")
            logger.info(classification_report(eval_set[1], preds['predict']))

        elif self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0 and not hasattr(self, "class_names"): #regression report
            test_features = H2OFrame(pd.DataFrame(data_dictionary["test_features"]))
            preds=model.leader.predict(test_features).as_data_frame()
            eval_set = (pd.DataFrame(data_dictionary["test_features"]),
                        pd.DataFrame(data_dictionary["test_labels"].to_numpy()[:, 0]))
            r2_regression = r2_score(eval_set[1], preds['predict'])
            logger.info(f"-------------------- Regression R2 Score: {r2_regression} --------------------")

        return model.leader

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param  unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        dk.data_dictionary["prediction_features"] = filtered_df
        
        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)
       
        predictions = self.model.predict(H2OFrame(pd.DataFrame(dk.data_dictionary["prediction_features"])))
        pred_df = predictions.as_data_frame()

        if hasattr(self, "class_names"):
            pred_df.columns = dk.label_list + self.class_names
        else:
            pred_df.columns = dk.label_list

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
            
        dk.do_predict = outliers
        
        return (pred_df, dk.do_predict)
    
    def define_data_pipeline(self, threads=-1) -> Pipeline:
        ft_params = self.freqai_info["feature_parameters"]
        pipe_steps = [
            ('const', ds.VarianceThreshold(threshold=0)),
            ('scaler', SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),
            ('imputer', SKLearnWrapper(SimpleImputer(strategy='mean'))),
            ]
        
        if ft_params.get("one_sided_gaussian_smoothing", False):
            pipe_steps.append(('smooth', SKLearnWrapper(OneSidedGaussianSmoothing(
                window_size=ft_params.get('gaussian_smooting_window_size', 5)))))

        if ft_params.get("polynomial_features", False):
            pipe_steps.append(('poly', SKLearnWrapper(PolynomialFeatures(degree=ft_params.get('polyfeat_degree', 2)))))
        
        if ft_params.get("power_transform", False):
            pipe_steps.append(('power', SKLearnWrapper(PowerTransformer(method='yeo-johnson'))))

        if ft_params.get("principal_component_analysis", False):
            pipe_steps.append(('pca', ds.PCA(n_components=0.999)))
            pipe_steps.append(('post-pca-scaler',
                               SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))))

        if ft_params.get("use_SVM_to_remove_outliers", False):
            svm_params = ft_params.get(
                "svm_params", {"shuffle": False, "nu": 0.01})
            pipe_steps.append(('svm', ds.SVMOutlierExtractor(**svm_params)))

        di = ft_params.get("DI_threshold", 0)
        if di:
            pipe_steps.append(('di', ds.DissimilarityIndex(di_threshold=di, n_jobs=threads)))

        if ft_params.get("use_DBSCAN_to_remove_outliers", False):
            pipe_steps.append(('dbscan', ds.DBSCAN(n_jobs=threads)))

        sigma = ft_params.get('noise_standard_deviation', 0)
        
        if sigma:
            pipe_steps.append(('noise', ds.Noise(sigma=sigma)))
            
        k = ft_params.get('k_best_number_of_features', 0)
        if k:
            pipe_steps.append(('featselect', SelectKBestTransform(k=k)))

        return Pipeline(pipe_steps)

    def define_label_pipeline(self, threads=-1) -> Pipeline:

        label_pipeline = Pipeline([
            ('scaler', SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1))))
            ])

        return label_pipeline
    
    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(f"-------------------- Starting training {pair} --------------------")

        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(f"-------------------- Training on data from {start_date} to "
                    f"{end_date} --------------------")
        # split data into train/test data.
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)

        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                  dd["train_labels"],
                                                                  dd["train_weights"])

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (dd["test_features"],
             dd["test_labels"],
             dd["test_weights"]) = dk.feature_pipeline.transform(dd["test_features"],
                                                                 dd["test_labels"],
                                                                 dd["test_weights"])

        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(f"Training model on {len(dd['train_features'])} data points")

        model = self.fit(dd, dk)

        end_time = time()

        logger.info(f"-------------------- Done training {pair} "
                    f"({end_time - start_time:.2f} secs) --------------------")

        return model
    

    
from sklearn.feature_selection import SelectKBest, f_classif
from datasieve.transforms.base_transform import BaseTransform
from scipy.stats import norm
import logging
import numpy as np

class SelectKBestTransform(BaseTransform):
    """
    A SelectKBest feature selection transform that ensures the feature names
    are properly transformed and follow along with the X throughout the pipeline.
    """

    def __init__(self, **kwargs):
        self._skl: SelectKBest = SelectKBest(**kwargs)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        n_components = X.shape[1]
        self._skl.fit(X, y)

        # Assuming feature_list is provided and matches the columns in X
        if feature_list is not None:
            selected_features = [feature_list[i] for i in self._skl.get_support(indices=True)]
        else:
            selected_features = [f"Feature{i}" for i in range(X.shape[1]) if self._skl.get_support()[i]]

        self.feature_list = selected_features  # Update feature_list here
        logger.info(f"selected {len(self.feature_list)} features of {n_components}")
        return X, y, sample_weight, self.feature_list

    def transform(self, X, y=None, sample_weight=None, outlier_check=False, feature_list=None, **kwargs):
        X = self._skl.transform(X)
        return X, y, sample_weight, self.feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        raise NotImplementedError("Inverse transform is not implemented for SelectKBest")


from sklearn.base import BaseEstimator, TransformerMixin
class OneSidedGaussianSmoothing(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=5):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Create a one-sided Gaussian window
        window = norm(loc=0, scale=self.window_size).pdf(np.arange(-self.window_size, 0))
        window /= window.sum()

        # Apply the filter to each feature
        X_smooth = np.empty_like(X)
        for i in range(X.shape[1]):
            X_smooth[:, i] = np.convolve(X[:, i], window, mode='same')

        return X_smooth

# class OneSidedGaussianSmoothing(BaseTransform):
#     def __init__(self, **kwargs):
#         if kwargs.get('window_size') is None:
#             window_size=5
#             logging.warning("OneSidedGaussianSmoothing: gaussian_smooting_window_size not provided, using default value of 5")

#         self.window_size = kwargs.get('window_size')

#     def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
#         return X, y, sample_weight, feature_list

#     def transform(self, X, y=None, sample_weight=None, outlier_check=False, feature_list=None, **kwargs):
#         # Create a one-sided Gaussian window
#         window = norm(loc=0, scale=self.window_size).pdf(np.arange(-self.window_size, 0))
#         window /= window.sum()

#         # Apply the filter to each feature
#         X_smooth = np.empty_like(X)
#         for i in range(X.shape[1]):
#             X_smooth[:, i] = np.convolve(X[:, i], window, mode='same')

#         return X, y, sample_weight, feature_list
    
#     def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
#         raise NotImplementedError("Inverse transform is not implemented for OneSidedGaussianSmoothing")
    

class NumpyToDataFrame(BaseTransform):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return self

    def transform(self, X, y=None, sample_weight=None, outlier_check=False, feature_list=None, **kwargs):
        return pd.DataFrame(X, columns=self.columns)
    
    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return X.to_numpy()