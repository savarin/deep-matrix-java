
public class Model {

  private double learningRate;
  private int iterationCount;
  private boolean verbose;
  private int numLabels;
  public Matrix W;
  public Matrix B;

  /**
   * Neural network-based predictive model class.
   *
   * @param learningRate Step size at each iteration.
   * @param iterationCount Number of epochs for training.
   * @param verbose For logging to stdout.
   */
  public Model(double learningRate, int iterationCount, boolean verbose) {
    this.learningRate = learningRate;
    this.iterationCount = iterationCount;
    this.verbose = verbose;
  }

  /**
   * Fits model to training data.
   *
   * @param trainX Training features matrix.
   * @param trainY Training labels matrix.
   */
  public void fit(Matrix trainX, Matrix trainY) throws Exception {
    Matrix[] results =
        Optimizers.parallel(trainX, trainY, this.learningRate, this.iterationCount, this.verbose);
    this.W = results[0];
    this.B = results[1];
    this.numLabels = trainY.unique();
  }

  /**
   * Predicts class probabilities of test data.
   *
   * @param testX Test features matrix.
   */
  public Matrix probabilities(Matrix testX) {
    int numRows = testX.shape()[0];
    int numColumns = testX.shape()[1];
    Matrix values = new Matrix(numRows, this.numLabels);

    for (int i = 0; i < numRows; i++) {
      Matrix rowData = testX.row(i).transpose();

      Matrix result = this.W.times(rowData).plus(this.B);
      Matrix softmaxResult = result.softmax();

      for (int j = 0; j < this.numLabels; j++) {
        values.entries[i][j] = softmaxResult.entries[j][0];
      }
    }

    return values;
  }

  /**
   * Predicts class labels of test data.
   *
   * @param testX Test features matrix.
   */
  public Matrix predict(Matrix testX) {
    int numRows = testX.shape()[0];
    int numColumns = testX.shape()[1];
    Matrix predictionY = new Matrix(numRows, 1);

    Matrix values = probabilities(testX);

    for (int i = 0; i < numRows; i++) {
      Matrix rowData = values.row(i).transpose();
      predictionY.entries[i][0] = (double) rowData.argmax()[0];
    }

    return predictionY;
  }
}
