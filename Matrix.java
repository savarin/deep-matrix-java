import java.util.Random;


public class Matrix {
  private int m;
  private int n;
  public double[][] entries;

  public Matrix(int m, int n) {
    this.m = m;
    this.n = n;
    entries = new double[m][n];
  }

  public Matrix(double[][] entries) {
    m = entries.length;
    n = entries[0].length;
    this.entries = new double[m][n];

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        this.entries[i][j] = entries[i][j];
      }
    }
  }
  
  public static Matrix identity(int n, double scalar) {
    Matrix I = new Matrix(n, n);

    for (int i=0; i<n; i++) {
      I.entries[i][i] = scalar;
    }

    return I;
  }  

  public static Matrix random(int m, int n) {
    Matrix R = new Matrix(m, n);
    Random random = new Random();

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        R.entries[i][j] = random.nextGaussian() * 0.01;
      }
    }

    return R;
  }

  public Matrix plus(Matrix B) {
    Matrix A = this;
    Matrix C = new Matrix(A.m, A.n);

    if (A.n != B.n || A.m != B.m) throw new RuntimeException("Improper matrix shape.");

    for (int i=0; i<A.m; i++) {
      for (int j=0; j<A.n; j++) {
        C.entries[i][j] = A.entries[i][j] + B.entries[i][j];
      }
    }

    return C;
  }

  public Matrix minus(Matrix B) {
    Matrix A = this;
    Matrix C = new Matrix(A.m, A.n);

    if (A.n != B.n || A.m != B.m) throw new RuntimeException("Improper matrix shape.");

    for (int i=0; i<A.m; i++) {
      for (int j=0; j<A.n; j++) {
        C.entries[i][j] = A.entries[i][j] - B.entries[i][j];
      }
    }

    return C;
  }

  public Matrix times(Matrix B) {
    Matrix A = this;
    Matrix C = new Matrix(A.m, B.n);
    
    if (A.n != B.m) throw new RuntimeException("Improper matrix shape.");

    for (int i=0; i<A.m; i++) {
      for (int j=0; j<B.n; j++) {
        for (int k=0; k<A.n; k++) {
          C.entries[i][j] += A.entries[i][k] * B.entries[k][j];
        }
      }
    }

    return C;
  }

  public Matrix transpose() {
    Matrix C = new Matrix(n, m);

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        C.entries[j][i] = entries[i][j];
      }
    }

    return C;
  }

  public Matrix softmax() {
    double expEntry;
    double expTotal = 0;
    Matrix C = new Matrix(m, n);

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        expEntry = Math.exp(entries[i][j]);
        expTotal += expEntry;
        C.entries[i][j] = expEntry;
      }
    }

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {      
        C.entries[i][j] = C.entries[i][j] / expTotal;
      }
    }

    return C;
  }

  public int[] argmax() {
    double maxEntry = entries[0][0];    
    int[] maxIndex = {0, 0};

    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        if (entries[i][j] > maxEntry) {
          maxIndex[0] = i;
          maxIndex[1] = j;
          maxEntry = entries[i][j];
        }
      }
    }

    return maxIndex;
  }

  public void show() {
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        System.out.printf("%.6f ", entries[i][j]);
      }
      System.out.println();
    }
  }

  public void shape() {
    System.out.printf("(%d, %d)", m, n);
    System.out.println();
  }
}