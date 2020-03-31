from keras.models import model_from_json 

PATH = "./uploads/"

def load(model_name, weight_name):
    
    model_name = PATH + model_name
    weight_name = PATH + weight_name

    try:
        json_file = open(model_name, "r")
        loaded_model_json = json_file.read() 
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print("{} : Model Loaded".format(model_name))
        try:
            loaded_model.load_weights(weight_name)
            print("{} : Loaded weights from disk".format(weight_name))
            return 0
        except:
            print("{} : Error Weight Load".format(weight_name))
            return 1
            #Weight Load Exceptions

    except:
        print("{} : Error Model Load".format(model_name))
        return 2
        #Model Load Exceptions