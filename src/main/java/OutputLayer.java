
import java.util.*;
import java.io.Serializable;

public class OutputLayer extends HiddenLayer implements Serializable {
    private int calculation = -1;
    private List<Double> activations;

    public OutputLayer(List<List<Double>> weights, List<Double> biases) {
        super(weights, biases);
    }

    public List<Double> calculate(List<Double> activations, boolean reset) {
        List<Double> res = super.calculate(activations, reset);
        int val = 0;
        for(int i = 0; i < res.size(); i++) {
            if(res.get(val) < res.get(i)) {
                val = i;
            }
        }
        calculation = val;
        this.activations = res;
        return res;
    }

    public int getVal() {
        return calculation;
    }

    public double getCost(int correct) {
        double cost = 0.0;
        for(int i = 0; i < activations.size(); i++) {
            if(i == correct) {
                cost += (activations.get(i) - 1) * (activations.get(i) - 1);
            } else {
                cost += activations.get(i) * activations.get(i);
            }
        }
        return cost;
    }
}
