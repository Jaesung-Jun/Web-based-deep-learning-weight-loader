from keras.models import model_from_json 

MODEL_PATH = "./uploads/"
WEIGHT_PATH = "./uploads/"

def load(model_name, weight_name):
    model_name = MODEL_PATH + model_name
    weight_name = WEIGHT_PATH + weight_name
    print(model_name)
    print(weight_name)

    try:
        json_file = open(model_name, "r")
        loaded_model_json = json_file.read() 
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print("Model Loaded")
        try:
            loaded_model.load_weights(weight_name)
            print("Loaded model from disk")
        except:
            print("Error Weight Load")
            pass
            #Weight Load Exceptions

    except:
        print("Error Model Load")
        pass
        #Model Load Exceptions