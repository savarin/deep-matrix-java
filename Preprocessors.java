
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class PreProcessing {

  /**
   * Reads data from a CSV file. Labels are in the left-most column, with
   * features in the remaining columns.
   * @param fileName Path of CSV file.
   * @return Matrix[] Features matrix and label matrix.
   */
  public static Matrix[] readCSV(String fileName) throws Exception {
    BufferedReader inputData = new BufferedReader(new FileReader(new File(fileName)));
    List<List<String>> stringData = new ArrayList<List<String>>();
    String line = "";

    while ((line = inputData.readLine()) != null) {
      stringData.add(Arrays.asList(line.split(",")));
    }

    int numRows = stringData.size();
    int numColumns = stringData.get(0).size();

    double[][] labels = new double[numRows][1];
    double[][] features = new double[numRows][numColumns-1];

    for (int i=0; i<numRows; i++) {
      labels[i][0] = Double.parseDouble(stringData.get(i).get(0));

      for (int j=1; j<numColumns; j++) {
        features[i][j-1] = Double.parseDouble(stringData.get(i).get(j));
      }
    }

    Matrix X = new Matrix(features);
    Matrix Y = new Matrix(labels);

    Matrix[] rawData = new Matrix[2];
    rawData[0] = X;
    rawData[1] = Y;

    return rawData;
  }

  /**
   * Scales matrix such that each column has mean 0 and standard deviation 1.
   * @param X Features matrix.
   * @return Matrix Scaled features matrix.
   */
  public static Matrix scaleFeatures(Matrix X) {
    int numRows = X.entries.length;
    int numColumns = X.entries[0].length;

    double[] columnMean = new double[numColumns];
    double[] columnStd = new double[numColumns];

    for (int j=0; j<numColumns; j++) {
      double columnSum = 0.0;
      double columnSquaredSum = 0.0;

      for (int i=0; i<numRows; i++) {
        columnSum += X.entries[i][j];
        columnSquaredSum += X.entries[i][j] * X.entries[i][j];
      }

      columnMean[j] = columnSum / (double) numRows;
      columnStd[j] = Math.sqrt(columnSquaredSum / (double) numRows - columnMean[j] * columnMean[j]);
    }

    for (int j=0; j<numColumns; j++) {
      double scaleMean = columnMean[j];
      double scaleStd = columnStd[j];

      for (int i=0; i<numRows; i++) {
        X.entries[i][j] = (X.entries[i][j] - scaleMean) / scaleStd;
      }
    }

    return X;
  }

  /**
   * Splits matrices into train and test matrices.
   * @param X Features matrix.
   * @param Y Label matrix.
   * @return Matrix[] Two features matrices and two label matrices.
   */
  public static Matrix[] trainTestSplit(Matrix X, Matrix Y, double trainSize) {
    int numRows = X.entries.length;
    int numColumns = X.entries[0].length;

    int numRows1 = (int) (numRows * trainSize);
    double[][] labels1 = new double[numRows1][1];
    double[][] features1 = new double[numRows1][numColumns];

    for (int i=0; i<numRows1; i++) {
      labels1[i][0] = Y.entries[i][0];

      for (int j=0; j<numColumns; j++) {
        features1[i][j] = X.entries[i][j];
      }
    }

    int numRows2 = numRows - numRows1;
    double[][] labels2 = new double[numRows2][1];
    double[][] features2 = new double[numRows2][numColumns];

    for (int i=0; i<numRows2; i++) {
      labels2[i][0] = Y.entries[numRows1+i][0];

      for (int j=0; j<numColumns; j++) {
        features2[i][j] = X.entries[numRows1+i][j];
      }
    }

    Matrix X1 = new Matrix(features1);
    Matrix Y1 = new Matrix(labels1);
    Matrix X2 = new Matrix(features2);
    Matrix Y2 = new Matrix(labels2);

    Matrix[] splitData = new Matrix[4];
    splitData[0] = X1;
    splitData[1] = Y1;
    splitData[2] = X2;
    splitData[3] = Y2;

    return splitData;
  }

  /**
   * Splits matrices into two by row indices.
   * @param X Features matrix.
   * @param Y Label matrix.
   * @return Matrix[] Two features matrices and two label matrices.
   */
  public static Matrix[] evenOddSplit(Matrix X, Matrix Y) {
    int numRows = X.entries.length;
    int numColumns = X.entries[0].length;

    int numRows1 = numRows / 2 + numRows % 2;
    double[][] labels1 = new double[numRows1][1];
    double[][] features1 = new double[numRows1][numColumns];

    // All rows with even-numbered indices
    for (int i=0; i<numRows; i+=2) {
      int rowIndex = i / 2;
      labels1[rowIndex][0] = Y.entries[i][0];

      for (int j=0; j<numColumns; j++) {
        features1[rowIndex][j] = X.entries[i][j];
      }
    }

    int numRows2 = numRows / 2;
    double[][] labels2 = new double[numRows2][1];
    double[][] features2 = new double[numRows2][numColumns];

    // All rows with odd-numbered indices
    for (int i=1; i<numRows; i+=2) {
      int rowIndex = (i-1) / 2;
      labels2[rowIndex][0] = Y.entries[i][0];

      for (int j=1; j<numColumns; j++) {
        features2[rowIndex][j] = X.entries[i][j];
      }
    }

    Matrix X1 = new Matrix(features1);
    Matrix Y1 = new Matrix(labels1);
    Matrix X2 = new Matrix(features2);
    Matrix Y2 = new Matrix(labels2);

    Matrix[] splitData = new Matrix[4];
    splitData[0] = X1;
    splitData[1] = Y1;
    splitData[2] = X2;
    splitData[3] = Y2;

    return splitData;
  }
}
