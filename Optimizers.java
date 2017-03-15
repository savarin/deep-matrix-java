
public class Optimizers implements Runnable {

  private Matrix X;
  private Matrix Y;
  private double learningRate;
  private int numRows;
  private int numColumns;
  private int numLabels;  
  public Matrix W;
  public Matrix B;

  public Optimizers(Matrix X, Matrix Y, double learningRate) {
    this.X = X;
    this.Y = Y;
    this.learningRate = learningRate;

    this.numRows = X.entries.length;
    this.numColumns = X.entries[0].length;
    this.numLabels = Y.unique();

    this.W = Matrix.random(this.numLabels, this.numColumns, 0.001);
    this.B = Matrix.random(this.numLabels, 1, 0.001);
  }

  public void run() {
    double loss = 0;

    for (int i=0; i<this.numRows; i++) {
      int label = (int) this.Y.entries[i][0];

      Matrix rowData = new Matrix(this.numColumns, 1);
      for (int j=0; j<this.numColumns; j++) {
        rowData.entries[j][0] = this.X.entries[i][j];
      }

      Matrix result = this.W.times(rowData).plus(this.B);
      Matrix normResult = result.softmax();

      int labelIndex = normResult.argmax()[0];
      loss += -Math.log(normResult.entries[labelIndex][0]);
      System.out.println(java.lang.Thread.currentThread().getName() + " "
          + loss / (double) i);

      Matrix labelOneHot = new Matrix(this.numLabels, 1);
      labelOneHot.entries[label][0] = 1.0;
      result = result.minus(labelOneHot);

      Matrix columnData = rowData.transpose();
      Matrix gradient = result.times(columnData);

      Matrix learningRateMatrix = Matrix.diagonal(this.numLabels, this.learningRate);
      gradient = learningRateMatrix.times(gradient);
      this.W = this.W.minus(gradient);
    }
  }

  public static Matrix[] parallelOptimization(Matrix X, Matrix Y, double 
      learningRate, int iterationCount) throws Exception {
    Matrix[] evenOddData = PreProcessing.evenOddSplit(X, Y);
    Matrix X1 = evenOddData[0];
    Matrix Y1 = evenOddData[1];
    Matrix X2 = evenOddData[2];
    Matrix Y2 = evenOddData[3];

    OptimizationProcesses p1 = new OptimizationProcesses(X1, Y1, learningRate);
    OptimizationProcesses p2 = new OptimizationProcesses(X2, Y2, learningRate);

    for (int i=0; i<iterationCount; i++) {
      java.lang.Thread t1 = new java.lang.Thread(p1);
      java.lang.Thread t2 = new java.lang.Thread(p1);

      t1.start();
      t2.start();

      t1.join();
      t2.join();  
    }

    Matrix W = p1.W.plus(p2.W);
    Matrix B = p1.B.plus(p2.B);
    Matrix divisor = Matrix.diagonal(W.entries.length, 0.5);
    W = divisor.times(W);
    B = divisor.times(B);
    
    Matrix[] results = new Matrix[2];
    results[0] = W;
    results[1] = B;

    return results;
  }

  /**
   * Naive model training by selecting the weights-bias matrix pair that 
   * produces the smallest lost.
   */
  public void naiveOptimization() {
    double minLoss = 1000.0;

    for (int k=0; k<1000; k++) {
      Matrix randomW = Matrix.random(this.numLabels, this.numColumns, 0.001);
      Matrix randomB = Matrix.random(this.numLabels, 1, 0.001);

      double loss = 0;

      for (int i=0; i<this.numRows; i++) {
        int label = (int) this.Y.entries[i][0];

        double[][] features = new double[this.numColumns][1];
        for (int j=0; j<this.numColumns; j++) {
          features[j][0] = this.X.entries[i][j];
        }

        Matrix rowData = new Matrix(features);

        Matrix result = randomW.times(rowData).plus(randomB);
        Matrix normResult = result.softmax();

        int labelIndex = normResult.argmax()[0];
        loss += -Math.log(normResult.entries[labelIndex][0]);
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

  public static void naiveTest() throws Exception {
    Matrix[] rawData = PreProcessing.readCSV("data.csv");
    Matrix X = PreProcessing.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    OptimizationProcesses p1 = new OptimizationProcesses(X, Y, 0.001);
    p1.naiveOptimization();
  }

  public static void gradientTest() throws Exception {
    Matrix[] rawData = PreProcessing.readCSV("data.csv");
    Matrix X = PreProcessing.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    OptimizationProcesses p1 = new OptimizationProcesses(X, Y, 0.001);
    java.lang.Thread t1 = new java.lang.Thread(p1);
    t1.start();
  }

  public static void parallelTest() throws Exception {
    Matrix[] rawData = PreProcessing.readCSV("data.csv");
    Matrix X = PreProcessing.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    Matrix[] results = parallelOptimization(X, Y, 0.001, 5);
    results[0].show();
    results[1].show();
  }  

  public static void main(String[] args) throws Exception {
    naiveTest();
    // gradientTest();
    // parallelTest();
  }
}









