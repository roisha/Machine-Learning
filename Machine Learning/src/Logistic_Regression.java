import java.util.List;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;


/**
 * Logistic regression implementation 
 */
public class Logistic_Regression {

	private double[] weights;
	private double learning_rate;
	private int iterations = 2000;

	public Logistic_Regression(int n) {
		learning_rate = 0.0001;
		weights = new double[n];
	}


	public static void main(String[] args) throws FileNotFoundException {

		List<vector> vectors = read_inputs_and_labels("data.txt");

		Logistic_Regression logistic_regression = new Logistic_Regression(5);

		logistic_regression.train(vectors);
		
		int[] x1 = new int[5] , x2 = new int[5];
		
		for (int i = 0; i < 5 ; i ++) {
			
			x1[i] = (int) (Math.random() * 3 );
		
			x2[i] = (int) (Math.random() * 3 ); 
			
		}

	

		System.out.println("prob(1|x) = " + logistic_regression.classification(x1));

		System.out.println("prob(1|x2) = " + logistic_regression.classification(x2));

	}

	public void train(List<vector> instances) {

		for (int i = 0; i < iterations; i++) {

			for (int j = 0; j < instances.size(); j ++) {

				int[] inputs = instances.get(j).inputs;
				double predicted = classification(inputs);
				int label = instances.get(j).given_label;

				for (int k = 0; k < weights.length; k ++) {

					weights[k] = weights[k] + learning_rate * (label - predicted) * inputs[k];
				}

			}
		}
	}

	private static double sigmoid_function(double z) {
		return 1.0 / (1.0 + Math.exp(- z));
	}

	private double classification(int[] x) {

		double reg_function_result = .0;

		for (int i = 0; i < weights.length; i ++)  {

			reg_function_result += weights[i] * x[i];

		}

		return sigmoid_function(reg_function_result);
	}

	public static class vector {
		public int given_label;
		public int[] inputs;

		public vector(int label, int[] inputs) {
			this.given_label = label;
			this.inputs = inputs;
		}
	}

	public static List<vector> read_inputs_and_labels(String file) throws FileNotFoundException {
		List<vector> dataset = new ArrayList<vector>();
		Scanner scanner = null;
		try {
			scanner = new Scanner(new File(file));
			while(scanner.hasNextLine()) {
				String line = scanner.nextLine();
				if (line.startsWith("#")) {
					continue;
				}
				String[] columns = line.split("\\s+");
				int i = 1;
				int[] data = new int[columns.length - 2];

				for (i = 1; i < columns.length - 1; i ++) {

					data[i - 1] = Integer.parseInt(columns[i]);
				}

				int label = Integer.parseInt(columns[i]);
				vector input_and_label = new vector(label, data);
				dataset.add(input_and_label);
			}
		} finally {
			if (scanner != null)
				scanner.close();
		}
		return dataset;
	}

}