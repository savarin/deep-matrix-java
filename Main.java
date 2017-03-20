
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
  public static void main(String[] args) throws Exception {
    String fileName = "data.csv";
    boolean test = false;
    boolean verbose = false;

    double learningRate = 0.001;
    double dropoutRate = 0.0;
    int iterationCount = 10;

    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.read(fileName);

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scale(rawData[0]);
    Matrix Y = rawData[1];

    if (test) {
      BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in), 1);
      System.out.println(
          "Please specify 1 for naive optimizer, 2 for gradient descent or 3 for parallel gradient descent:");
      int choice = Integer.parseInt(stdin.readLine());

      switch (choice) {
        case 1:
          Optimizers p1 = new Optimizers(X, Y, learningRate, dropoutRate, true);
          p1.naive();
          break;
        case 2:
          Optimizers p2 = new Optimizers(X, Y, learningRate, dropoutRate, true);
          java.lang.Thread t = new java.lang.Thread(p2);
          t.start();
          t.join();
          break;
        case 3:
          Matrix[] results =
              Optimizers.parallel(X, Y, learningRate, dropoutRate, iterationCount, true);
          break;
      }

      System.out.println("Test successful!");
      System.exit(0);
    }

    Matrix[] trainTestData = Preprocessors.split(X, Y, 0.2);
    Matrix trainX = trainTestData[0];
    Matrix trainY = trainTestData[1];
    Matrix testX = trainTestData[2];
    Matrix testY = trainTestData[3];

    System.out.println("Initializing model...");
    Model model = new Model(learningRate, dropoutRate, iterationCount, verbose);

    System.out.println("Training model...");
    model.fit(trainX, trainY);

    System.out.println("Making predictions...");
    Matrix predictionY = model.predict(testX);

    System.out.println("Evaluating predictions...");
    double accuracyScore = Metrics.accuracy(testY, predictionY);
    double precisionScore = Metrics.precision(testY, predictionY);
    double recallScore = Metrics.recall(testY, predictionY);

    System.out.printf("Accuracy : %.3f\n", accuracyScore);
    System.out.printf("Precision: %.3f\n", precisionScore);
    System.out.printf("Recall   : %.3f\n", recallScore);

    model.W.show();
  }
}
