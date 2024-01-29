import java.util.*;

public interface Layer {
    public List<Double> calculate(List<Double> activations, boolean reset);

    public int getCount();

    public List<List<Double>> getWeights();

    public List<Double> getBiases();
}
