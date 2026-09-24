import mlflow.pyfunc
import pickle

class PopularityModelWrapper(mlflow.pyfunc.PythonModel):

    def load_contex(self,context):

        with open(context.artifacts["popularity_model"],"rb") as f:
            self.model=pickle.load(f)


    def predict(self,context,model_input):

        if hasattr(model_input,"columns"):

            if 'N' in model_input.columns:
                N=int(model_input['N'].iloc[0])
            else:
                N=10
        else:
            N=10


        return self.model.recommend(N=N)


