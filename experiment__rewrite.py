from lester.benchmark.creditcard_dataprep import CreditcardDataprepTask
from lester.benchmark.yichun_dataprep import YichunDataprepTask
from lester.benchmark.amazonreviews_dataprep import AmazonreviewsDataprepTask
from lester.benchmark.creditcard_featurisation import CreditcardFeaturisationTask
from lester.benchmark.ldb_featurisation import LdbFeaturisationTask
from lester.benchmark.titanic_featurisation import TitanicFeaturisationTask
from lester.benchmark.sklearnlogreg_model import SklearnLogisticRegressionTransformationTask
from lester.benchmark.sklearnsvm_model import SklearnSVMTransformationTask
from lester.benchmark.sklearnmlp_model import SklearnMLPTransformationTask

from synthesised_code import *


if __name__ == "__main__":
    print('CreditcardDataprepTask...')
    creditcard_task = CreditcardDataprepTask()
    creditcard_task.evaluate_transformed_code(SYNTHESISED_CREDITCARD_DATAPREP_CODE)

    print('YichunDataprepTask...')
    yichun_task = YichunDataprepTask()
    yichun_task.evaluate_transformed_code(SYNTHESISED_YICHUN_DATAPREP_CODE)

    print('AmazonreviewsDataprepTask...')
    amazonreviews_task = AmazonreviewsDataprepTask()
    amazonreviews_task.evaluate_transformed_code(SYNTHESISED_AMAZONREVIEWS_DATAPREP_CODE)

    print('CreditcardFeaturisation...')
    creditcard_task = CreditcardFeaturisationTask()
    creditcard_task.evaluate_transformed_code(SYNTHESISED_CREDITCARD_FEATURISATION_CODE)

    print('LdbFeaturisation...')
    ldb_task = LdbFeaturisationTask()
    ldb_task.evaluate_transformed_code(SYNTHESISED_LDB_FEATURISATION_CODE)

    print('TitanicFeaturisation...')
    titanic_task = TitanicFeaturisationTask()
    titanic_task.evaluate_transformed_code(SYNTHESISED_TITANIC_FEATURISATION_CODE)

    print("SklearnLogisticRegression...")
    logregtask = SklearnLogisticRegressionTransformationTask()
    logregtask.evaluate_transformed_code(SYNTHESISED_SKLEARNLOGREG_CODE)

    print("SklearnSVM...")
    svmtask = SklearnSVMTransformationTask()
    svmtask.evaluate_transformed_code(SYNTHESISED_SKLEARNSVM_CODE)

    print("SklearnMLP...")
    mlptask = SklearnMLPTransformationTask()
    mlptask.evaluate_transformed_code(SYNTHESISED_SKLEARNMLP_CODE)
