import sys
sys.path.append('../../')

from lester.benchmark import DataprepCodeTransformationTask
from lester.benchmark import FeaturisationCodeTransformationTask
from lester.benchmark import ModelCodeTransformationTask

# Should the other methods also become variables?
# Originally I had them as @Property but in the rewrite/__init__.py file
# Sebastian only uses the @Property keyword to define the original_code attribute


class LLMDataprepTask(DataprepCodeTransformationTask):

    def __init__ (self, original_code, input_arg_names, input_schemas, output_columns):
        self._original_code = original_code
        self._input_arg_names = input_arg_names
        self._input_schemas = input_schemas
        self._output_columns = output_columns
        self._synthesized_code = None
        self._colour = "green"

    @property
    def original_code(self):
        return self._original_code
    
    @property
    def colour(self):
        return self._colour
    
    @property
    def synthesized_code(self):
        return self._synthesized_code
    
    def set_synthesized_code(self, new_synthesized_code):
        self._synthesized_code = new_synthesized_code

    def input_arg_names(self):
        return self._input_arg_names

    def input_schemas(self):
        return self._input_schemas

    def output_columns(self):
        return self._output_columns
    
    def run_manually_rewritten_code(self, params):
        raise NotImplementedError("This method is not implemented in this subclass.")

    def evaluate_transformed_code(self, transformed_code):
        raise NotImplementedError("This method is not implemented in this subclass.")
    


class LLMFeaturisationTask(FeaturisationCodeTransformationTask):

    def __init__ (self, original_code, input_schemas):
        self._original_code = original_code
        self._input_schema = input_schemas
        self._synthesized_code = None
        self._colour = "yellow"

    @property
    def original_code(self):
        return self._original_code
    
    @property
    def colour(self):
        return self._colour

    @property
    def input_schema(self):
        return self._input_schema
    
    @property
    def synthesized_code(self):
        return self._synthesized_code
    
    def set_synthesized_code(self, new_synthesized_code):
        self._synthesized_code = new_synthesized_code

    def evaluate_transformed_code(self, transformed_code):
        raise NotImplementedError("This method is not implemented in this subclass.")


class LLMModelCodeTransformationTask(ModelCodeTransformationTask):

    def __init__ (self, original_code):
        self._original_code = original_code
        self._synthesized_code = None
        self._colour = "red"

    @property
    def original_code(self):
        return self._original_code
    
    @property
    def colour(self):
        return self._colour
    
    @property
    def synthesized_code(self):
        return self._synthesized_code
    
    def set_synthesized_code(self, new_synthesized_code):
        self._synthesized_code = new_synthesized_code

    def evaluate_transformed_code(self, transformed_code):

        raise NotImplementedError("This method is not implemented in this subclass.")



    