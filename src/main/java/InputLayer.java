import java.util.*;
import java.io.Serializable;

public class InputLayer implements Serializable, Layer {
    private List<Double> calculation;
    private int count;

    public InputLayer(int count) {
        calculation = null;
        this.count = count;
    }

    public List<Double> calculate(List<Double> activations, boolean reset) {
        if(!reset && calculation != null) {
            return calculation;
        }
        calculation = activations;
        return calculation;
    }

    public int getCount() {
        return count;
    }

    public List<List<Double>> getWeights() {
        return null;
    }

    public List<Double> getBiases() {
        return null;
    }
}
