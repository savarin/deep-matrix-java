
public class Optimizers implements Runnable {

  private Matrix X;
  private Matrix Y;
  private double learningRate;
  private double dropoutRate;
  private boolean verbose;
  private int numRows;
  private int numColumns;
  private int numLabels;
  public Matrix W;
  public Matrix B;

  /**
   * Gradient descent optimizer class.
   *
   * @param X Features matrix.
   * @param Y Labels matrix.
   * @param learningRate Step size at each iteration.
   * @param verbose For logging to stdout.
   */
  public Optimizers(Matrix X, Matrix Y, double learningRate, double dropoutRate, boolean verbose) {
    this.X = X;
    this.Y = Y;
    this.learningRate = learningRate;
    this.dropoutRate = dropoutRate;
    this.verbose = verbose;

    this.numRows = X.shape()[0];
    this.numColumns = X.shape()[1];
    this.numLabels = Y.unique();

    this.W = Matrix.random(this.numLabels, this.numColumns, 0.001);
    this.B = Matrix.random(this.numLabels, 1, 0.001);
  }

  /** Thread for gradient descent update on weight matrix. */
  public void run() {
    double loss = 0;

    for (int i = 0; i < this.numRows; i++) {
      int label = (int) this.Y.entries[i][0];
      Matrix rowData = this.X.row(i).transpose();

      Matrix result = this.W.times(rowData).plus(this.B);

      if (this.verbose) {
        Matrix softmaxResult = result.softmax();
        int labelIndex = softmaxResult.argmax()[0];
        loss += -Math.log(softmaxResult.entries[labelIndex][0]);

        System.out.println(java.lang.Thread.currentThread().getName() + " " + loss / (double) i);
      }

      Matrix labelOneHot = new Matrix(this.numLabels, 1);
      labelOneHot.entries[label][0] = 1.0;
      result = result.relu().dropout(this.dropoutRate).minus(labelOneHot);

      Matrix columnData = rowData.transpose();
      Matrix gradient = result.times(columnData);

      Matrix alpha = Matrix.diagonal(this.numLabels, this.learningRate);
      gradient = alpha.times(gradient);
      this.W = this.W.minus(gradient);
    }
  }

  /**
   * Gradient descent implementation with parallel training.
   *
   * @param X Features matrix.
   * @param Y Labels matrix.
   * @param learningRate Step size at each iteration.
   * @param iterationCount Number of epochs for training.
   * @param verbose For logging to stdout.
   */
  public static Matrix[] parallel(
      Matrix X, Matrix Y, double learningRate, double dropoutRate, int iterationCount, boolean verbose)
      throws Exception {
    Matrix[] evenOddData = Preprocessors.bisect(X, Y);
    Matrix evenX = evenOddData[0];
    Matrix evenY = evenOddData[1];
    Matrix oddX = evenOddData[2];
    Matrix oddY = evenOddData[3];

    Optimizers p1 = new Optimizers(evenX, evenY, learningRate, dropoutRate, verbose);
    Optimizers p2 = new Optimizers(oddX, oddY, learningRate, dropoutRate, verbose);

    for (int i = 0; i < iterationCount; i++) {
      System.out.printf("Running iteration %d of %d...\n", i + 1, iterationCount);
      java.lang.Thread t1 = new java.lang.Thread(p1);
      java.lang.Thread t2 = new java.lang.Thread(p2);

      t1.start();
      t2.start();

      t1.join();
      t2.join();
    }

    Matrix W = p1.W.plus(p2.W);
    Matrix B = p1.B.plus(p2.B);
    Matrix divisor = Matrix.diagonal(W.shape()[0], 0.5);
    W = divisor.times(W);
    B = divisor.times(B);

    Matrix[] results = new Matrix[2];
    results[0] = W;
    results[1] = B;

    return results;
  }

  /** Naive model training by selecting the weights-bias matrix pair that produces the smallest loss. */
  public void naive() {
    double minLoss = 1000.0;

    for (int k = 0; k < 1000; k++) {
      Matrix randomW = Matrix.random(this.numLabels, this.numColumns, 0.001);
      Matrix randomB = Matrix.random(this.numLabels, 1, 0.001);

      double loss = 0;

      for (int i = 0; i < this.numRows; i++) {
        int label = (int) this.Y.entries[i][0];
        Matrix rowData = this.X.row(i).transpose();

        Matrix result = randomW.times(rowData).plus(randomB);
        result = result.softmax();

        int labelIndex = result.argmax()[0];
        loss += -Math.log(result.entries[labelIndex][0]);
      }

      loss = loss / numRows;

      if (loss < minLoss) {
        minLoss = loss;
        this.W = randomW;
        this.B = randomB;
        System.out.printf("loss %.9f loop %d", loss, k);
        System.out.println();
      }
    }
  }
}
