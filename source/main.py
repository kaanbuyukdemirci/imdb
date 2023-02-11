import data_processing as dp
import machine_learning as ml

def data_processing(batch_size = 10**6):
    """
    Run this function only once. It basically extracts data that we can use to train machine learning models. 
    The data extracted is saved in data\\processed directory.
    """
    dp.filter1(batch_size)
    print("filter1 is done!",end='r')
    dp.filter2(batch_size)
    print("filter2 is done!",end='r')
    dp.filter3(batch_size)
    print("filter3 is done!",end='r')
    dp.filter4(batch_size)
    print("filter4 is done!",end='r')
    dp.filter5(batch_size)
    print("filter5 is done!",end='r')
    dp.filter6(batch_size)
    print("filter6 is done!",end='r')
    dp.filter7(batch_size)
    print("filter7 is done!",end='r')
    dp.filter8(batch_size) # not optimized, and has a minimum requirement for batch size.
    print("filter8 is done!",end='r')
    dp.filter9(batch_size) # not optimized, and has a minimum requirement for batch size.
    print("filter9 is done!",end='r')
    dp.filter10() # has no batch size, can be added, but it is not really necessary for now.
    print("filter10 is done!",end='r')

def machine_learning():
    """
    Run it to train a linear regression and a neural network model.
    
    First run ml.prepare_data()
    Then run ml.linear_regression() if you want to train a linear regression model.
    Then run ml.neural_network() if you want to train a neural network model.
    """
    #ml.prepare_data()
    ml.linear_regression()
    ml.neural_network(startOver=True, learning_rate=0.001, 
                      lamb_regularization_parameter=0.001, 
                      batch_size=32, number_of_epochs=100)

if __name__ == "__main__":
    #data_processing()
    machine_learning()
