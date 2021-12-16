# Stock Index Prediction Using Frequency Decomposition and Neural Networks

Ensemble Empirical Mode Decomposition (Ensemble EMD) – Ensemble EMD is a developed version from EMD. One of the reasons why these techniques are powerful is that they do not assume the signal to be periodic or any pattern. In another word, the methods are completely based on the facts of the original signal. Both EMD and ensemble EMD will decompose the original signal into Intrinsic Mode Functions (IMFs). However, EMD suffers from mode mixing problems especially when dealing with complex signals. Simply speaking, some IMF with a specific frequency might include mixed signals with different (higher or lower) frequency. Ensemble EMD was proposed by Wu and Huang in 2009 to solve this drawback. Ensemble EMD can decompose any complex signals into IMFs, and it is intuitive and adaptive. By adding white noise, Ensemble EMD can separate scales naturally instead of intermittence test using a priori subjective criterion selection like EMD. Ensemble EMD would be used to decompose signals into IMFs. Several IMFs (the number of IMFs is based on the complexity of original signal) generated would be ordered from highest frequency to lowest frequency along with a residual subsequence indicating the overall trend of the original time series.

Long-short Term Memory (LSTM) – LSTM is a special kind of recurrent neural networks (RNN) that can manage the past sequential data and deal with long-term memory. It is a popular and powerful deep learning method in sequential modeling. Not like a simple RNN model which has the vanishing gradient issue, LSTM adds three gates to avoid the problem: the forget gates, the input gates, and the output gates: The forget gates can decide which information should be ignored; the input gates can tell if new data will be added into the memory thus it can update the cell status; the output gates can calculate the next hidden state. These three gates provide the LSTM model the ability to write, read, and reset continuously. Therefore, LSTM has the power to learn long- term dependencies.

(Multiple) Linear Regression – Linear regression is a simple machine learning model that determines the relationship between the response variable and the independent variables by fitting a linear equation to obtain coefficients. After decomposing the original signal by using Ensemble EMD, how to fuse the predictions separately done on each subsequence would be one of the key steps. Just simply summing up all the subsequences is not enough under this scenario, because during the experiment we have discovered different sequences can take up different weights in the original signal. Then, weighted sum is under our consideration. Thus, using linear regression as a weight assigning method determines the weights for each of our subsequences for the later fusion process.

Then, the hybrid method implemented, Ensemble EMD-LSTM method with weights generated from linear regression, takes advantages of above methods described. First,  decompose the time-series data into n IMFs using the Ensemble EMD model. Then, linear Regression is used on the subsequences (IMFs) to assign weights for each of them such that we can fuse them by using weighted sum later in the process. After that, apply LSTMs on each subsequence separately to get n separate predictions (n equals to the number of IMFs), and then fuse the predictions using the weights we get from the Linear Regression step. After fusion, final prediction is generated and then we can do performance evaluations using RMSE (root-mean-square error).
