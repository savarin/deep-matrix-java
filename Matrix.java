public class Matrix {
  
  public static void main(String[] args) {
    float [][] A = {{1, 2}, {2, 1}};
    float [][] B = {{1, 0}, {0, 1}};

    float [][] C = new float[2][2];

    for (int i=0; i<A.length; i++) {
      for (int j=0; j<B[0].length; j++) {
        for (int k=0; k<A[0].length; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    for (int i=0; i<C.length; i++) {
      for (int j=0; j<C[0].length; j++) {
        System.out.println(C[i][j]);
      }
    }
  }
}