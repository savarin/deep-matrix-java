
public class Model {

  private double learningRate;
  private int iterationCount;
  private int numLabels;
  public Matrix W;
  public Matrix B;

  public Model(double learningRate, int iterationCount) {
    this.learningRate = learningRate;
    this.iterationCount = iterationCount;
  }

  public void fit(Matrix trainX, Matrix trainY) throws Exception {
    Matrix[] results = OptimizationProcesses.parallelOptimization(trainX, 
        trainY, this.learningRate, this.iterationCount);
    this.W = results[0];
    this.B = results[1];
    this.numLabels = trainY.unique();
  }

  public Matrix predictProbability(Matrix testX) {
    int numRows = testX.entries.length;
    int numColumns = testX.entries[0].length;
    Matrix probabilities = new Matrix(numRows, this.numLabels);

    for (int i=0; i<numRows; i++) {
      Matrix rowData = new Matrix(numColumns, 1);
      for (int j=0; j<numColumns; j++) {
        rowData.entries[j][0] = testX.entries[i][j];
      }

      Matrix result = this.W.times(rowData).plus(this.B);
      Matrix normResult = result.softmax();

      for (int j=0; j<this.numLabels; j++) {
        probabilities.entries[i][j] = normResult.entries[j][0];
      }
    }

    return probabilities;
  }

  public Matrix predictClasses(Matrix testX) {
    int numRows = testX.entries.length;
    int numColumns = testX.entries[0].length;
    Matrix predictionY = new Matrix(numRows, 1);

    Matrix probabilities = predictProbability(testX);

    for (int i=0; i<numRows; i++) {
      Matrix rowData = new Matrix(numColumns, 1);
      for (int j=0; j<this.numLabels; j++) {
        rowData.entries[j][0] = probabilities.entries[i][j];
      }

      predictionY.entries[i][0] = (double) rowData.argmax()[0];
    }

    return predictionY;
  }

  public static void modelTest() throws Exception {
    Matrix[] rawData = PreProcessing.readCSV("data.csv");
    Matrix X = PreProcessing.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    Matrix[] trainTestData = PreProcessing.trainTestSplit(X, Y, 0.80);
    Matrix trainX = trainTestData[0];
    Matrix trainY = trainTestData[1];
    Matrix testX = trainTestData[2];
    Matrix testY = trainTestData[3];

    NeuralNetworkModel model = new NeuralNetworkModel(0.0001, 5);

    model.fit(trainX, trainY);
    Matrix predictionY = model.predictProbability(testX);

    System.out.println(PerformanceMetrics.accuracyScore(testY, predictionY));
  }  

  public static void main(String[] args) throws Exception {
    modelTest();
  }
}
