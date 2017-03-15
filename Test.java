
public class Test {

  public static void naiveTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.readCSV("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    System.out.println("Initializing model...");
    Optimizers p1 = new Optimizers(X, Y, 0.001);

    System.out.println("Training model...");
    p1.naiveOptimization();
  }

  public static void gradientTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.readCSV("data.csv");
    
    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    System.out.println("Initializing model...");
    Optimizers p1 = new Optimizers(X, Y, 0.001);

    System.out.println("Training model...");
    java.lang.Thread t1 = new java.lang.Thread(p1);
    t1.start();
  }

  public static void parallelTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.readCSV("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    System.out.println("Initializing model...");
    System.out.println("Training model...");
    Matrix[] results = Optimizers.parallelOptimization(X, Y, 0.001, 5);
  }

  public static void modelTest() throws Exception {
    System.out.println("Loading data...");
    Matrix[] rawData = Preprocessors.readCSV("data.csv");

    System.out.println("Preprocessing data...");
    Matrix X = Preprocessors.scaleFeatures(rawData[0]);
    Matrix Y = rawData[1];

    Matrix[] trainTestData = Preprocessors.trainTestSplit(X, Y, 0.80);
    Matrix trainX = trainTestData[0];
    Matrix trainY = trainTestData[1];
    Matrix testX = trainTestData[2];
    Matrix testY = trainTestData[3];

    System.out.println("Initializing model...");
    Model model = new Model(0.001, 5);

    System.out.println("Training model...");
    model.fit(trainX, trainY);

    System.out.println("Making predictions...");
    Matrix predictionY = model.predictClasses(testX);

    System.out.println("Evaluating performance...");
    double accuracyScore = Metrics.accuracy(testY, predictionY);
    double precisionScore = Metrics.precision(testY, predictionY);
    double recallScore = Metrics.recall(testY, predictionY);

    System.out.printf("Accuracy: %.3f\n", accuracyScore);
    System.out.printf("Precision: %.3f\n", precisionScore);
    System.out.printf("Recall: %.3f\n", recallScore);
  }

  public static void main(String[] args) throws Exception {
    // naiveTest();
    // gradientTest();
    // parallelTest();
    modelTest();
  }
}
