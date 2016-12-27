import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class Model {

  public static List<List<String>> readCSV(String fileName)
      throws FileNotFoundException, IOException {
    BufferedReader inputData = new BufferedReader(new FileReader(new File(fileName)));
    List<List<String>> data = new ArrayList<List<String>>();
    String line = "";

    while ((line = inputData.readLine()) != null) {
      data.add(Arrays.asList(line.split(",")));
    }

    return data;
  }

  public static void testSingleRow() {
    double[][] x = {{0.83290956}, {0.74926865}, {-0.61259594}, {-0.51933199}};
    Matrix X = new Matrix(x);

    double[][] w = {{0.00262025, 0.00158684, 0.00278127, 0.00459317}, {0.00321001, 0.00518393, 0.00261943, 0.00976085}};
    Matrix W = new Matrix(w);

    double[][] b = {{0.00732815}, {0.00115274}};
    Matrix B = new Matrix(b);

    Matrix result = new Matrix(2, 1);
    result = W.times(X).plus(B);
    result.show();

    result = result.softmax();
    result.show();

    int labelIndex = result.argmax()[0];

    System.out.println(-Math.log(result.entries[labelIndex][0]));
  }

  public static void testAllRows()
      throws FileNotFoundException, IOException {
    List<List<String>> data = readCSV("data.csv");

    int numRows = data.size();
    int numColumns = data.get(0).size();

    double[][] w = {{0.00262025, 0.00158684, 0.00278127, 0.00459317},
        {0.00321001, 0.00518393, 0.00261943, 0.00976085}};
    Matrix W = new Matrix(w);

    double[][] b = {{0.00732815}, {0.00115274}};
    Matrix B = new Matrix(b);

    double loss = 0;

    for (int i=1; i<numRows; i++) {
      int label = Integer.parseInt(data.get(i).get(0));

      double[][] features = new double[numColumns-1][1];
      for (int j=1; j<numColumns; j++) {
        features[j-1][0] = Double.parseDouble(data.get(i).get(j));
      }

      Matrix X = new Matrix(features);

      Matrix result = new Matrix(2, 1);
      result = W.times(X).plus(B);
      result = result.softmax();

      int labelIndex = result.argmax()[0];

      loss += -Math.log(result.entries[labelIndex][0]);
    }

    System.out.println(loss / numRows);
  }

  public static void testNaiveOptimization()
      throws FileNotFoundException, IOException {
    List<List<String>> data = readCSV("data.csv");

    int numRows = data.size();
    int numColumns = data.get(0).size();

    double minLoss = 1000.0;
    Matrix bestWeights = new Matrix(2, 4);
    Matrix bestBias = new Matrix(2, 1);

    for (int i=0; i<1000; i++) {
      Matrix W = new Matrix(2, 4);
      W = Matrix.random(2, 4);

      Matrix B = new Matrix(2, 1);
      B = Matrix.random(2, 1);

      double loss = 0;

      for (int j=1; j<numRows; j++) {
        int label = Integer.parseInt(data.get(j).get(0));

        double[][] features = new double[numColumns-1][1];
        for (int k=1; k<numColumns; k++) {
          features[k-1][0] = Double.parseDouble(data.get(j).get(k));
        }

        Matrix X = new Matrix(features);

        Matrix result = new Matrix(2, 1);
        result = W.times(X).plus(B);
        result = result.softmax();

        int labelIndex = result.argmax()[0];

        loss += -Math.log(result.entries[labelIndex][0]);
      }

      loss = loss / numRows;

      if (loss < minLoss) {
        minLoss = loss;
        bestWeights = W;
        bestBias = B;
        System.out.printf("loss %.9f loop %d", loss, i);
        System.out.println();
      }
    }
  }

  public static void main(String[] args)
    throws FileNotFoundException, IOException {

    testSingleRow();
    testAllRows();
    testNaiveOptimization();
  }
}


