
public class Test {

  public static void naiveTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.read("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scale(rawData[0]);
    Matrix Y = rawData[1];

    System.out.println("Initializing model...");
    Optimizers p1 = new Optimizers(X, Y, 0.001, 0.2, true);

    System.out.println("Training model...");
    p1.naive();
  }

  public static void gradientTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.read("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scale(rawData[0]);
    Matrix Y = rawData[1];

    System.out.println("Initializing model...");
    Optimizers p1 = new Optimizers(X, Y, 0.001, 0.2, true);

    System.out.println("Training model...");
    java.lang.Thread t1 = new java.lang.Thread(p1);
    t1.start();
  }

  public static void parallelTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.read("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scale(rawData[0]);
    Matrix Y = rawData[1];

    System.out.println("Initializing model...");
    System.out.println("Training model...");
    Matrix[] results = Optimizers.parallel(X, Y, 0.001, 0.2, 5, true);
  }

  public static void modelTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.read("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scale(rawData[0]);
    Matrix Y = rawData[1];

    Matrix[] trainTestData = Preprocessors.split(X, Y, 0.20);
    Matrix trainX = trainTestData[0];
    Matrix trainY = trainTestData[1];
    Matrix testX = trainTestData[2];
    Matrix testY = trainTestData[3];

    System.out.println("Initializing model...");
    Model model = new Model(0.001, 0.2, 10, false);

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
  }

  public static void main(String[] args) throws Exception {
    // naiveTest();
    // gradientTest();
    // parallelTest();
    modelTest();
  }
}
