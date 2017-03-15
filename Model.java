
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
    Matrix[] results =
        Optimizers.parallelOptimization(trainX, trainY, this.learningRate, this.iterationCount);
    this.W = results[0];
    this.B = results[1];
    this.numLabels = trainY.unique();
  }

  public Matrix predictProbability(Matrix testX) {
    int numRows = testX.shape()[0];
    int numColumns = testX.shape()[1];
    Matrix probabilities = new Matrix(numRows, this.numLabels);

    for (int i = 0; i < numRows; i++) {
      Matrix rowData = testX.selectRow(i).transpose();

      Matrix result = this.W.times(rowData).plus(this.B);
      Matrix normResult = result.softmax();

      for (int j = 0; j < this.numLabels; j++) {
        probabilities.entries[i][j] = normResult.entries[j][0];
      }
    }

    return probabilities;
  }

  public Matrix predictClasses(Matrix testX) {
    int numRows = testX.shape()[0];
    int numColumns = testX.shape()[1];
    Matrix predictionY = new Matrix(numRows, 1);

    Matrix probabilities = predictProbability(testX);

    for (int i = 0; i < numRows; i++) {
      Matrix rowData = probabilities.selectRow(i).transpose();
      predictionY.entries[i][0] = (double) rowData.argmax()[0];
    }

    return predictionY;
  }
}
