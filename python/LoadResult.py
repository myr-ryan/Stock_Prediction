from TSLibrary import StockPredictionSystem as spSystem

#obj = spSystem.load_experiment('./result/LSTMRegressor_all/experiment.txt')

#for overnight
#obj = spSystem.load_experiment('./result/LSTMRegressor_all_overnight/experiment_overnight.txt')

#combine
obj = spSystem.load_experiment('./result/LSTMRegressor_all_combine/experiment_combine.txt')

print(obj)
