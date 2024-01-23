import java.util.*;
import java.io.Serializable;

public class HiddenLayer implements Serializable, Layer {
    private List<List<Double>> weights; // neuron x weights, count rows, prevCount cols
    private List<Double> biases; // biases for each neuron in count
    private int count;
    private int prevCount;
    private List<Double> calculation;

    public HiddenLayer(List<List<Double>> weights, List<Double> biases) {
        if(weights != null && biases != null && weights.size() != biases.size()) {
            throw new RuntimeException("Mismatched Dimensions");
        }
        count = biases.size();
        prevCount = weights.get(0).size();
        this.weights = weights;
        this.biases = biases;
        calculation = null;
    }

    public List<Double> getBiases() {
        return biases;
    }

    public List<List<Double>> getWeights() {
        return weights;
    }

    // return list of current layer activation values
    public List<Double> calculate(List<Double> activations, boolean reset) {
        if(!reset && calculation != null) {
            return calculation;
        }
        if(activations.size() != prevCount) {
            throw new RuntimeException("Mismatched Dimensions");
        }
        ArrayList<Double> res = new ArrayList<>();
        for(int i = 0; i < count; i++) {
            double d = 0;
            double bias = biases.get(i);
            for(int j = 0; j < prevCount; j++) {
                d += weights.get(i).get(j) * activations.get(j);
            }
            d += bias;
            res.add(1.0 / (1 + Math.exp(-1 * d)));
            //System.out.println(d);
        }
        //System.out.println(activations);
        //System.out.println(calculation);
        calculation = res;
        return calculation;
    }

    public int getCount() {
        return count;
    }
}