import java.util.*;
import java.io.*;

public class Network implements Serializable {
    private List<Layer> layers;
    private OutputLayer outputLayer;

    public Network(int[] counts) {
        if(counts.length < 2) {
            throw new RuntimeException("Invalid Layer Count");
        }
        layers = new ArrayList<Layer>();
        layers.add(new InputLayer(counts[0]));
        for(int i = 1; i < counts.length - 1; i++) {
            HiddenLayer midlayer = new HiddenLayer(randomWeights(counts[i], counts[i - 1]), setupBiases(counts[i]));
            layers.add(midlayer);
        }
        outputLayer = new OutputLayer(randomWeights(counts[counts.length - 1], counts[counts.length - 2]), setupBiases(counts[counts.length - 1]));
        layers.add(outputLayer);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public double getCost(int correct) {
        return outputLayer.getCost(correct);
    }

    public int getVal(List<Integer> pixels) {
        if(pixels.size() != layers.get(0).getCount()) {
            throw new RuntimeException("Invalid Pixel Count: Expected " + layers.get(0).getCount() + " but got " + pixels.size());
        }
        List<Double> activations = new ArrayList<Double>();
        for(int i : pixels) {
            if(i < 0 || i > 255) {
                throw new RuntimeException("Invalid Pixel Value: " + i);
            }
            activations.add(i / 255.0);
        }
        //System.out.println(activations);
        for(Layer l : layers) {
            activations = l.calculate(activations, true);
        }
        return outputLayer.getVal();
    }

    private List<List<Double>> randomWeights(int count, int prevCount) {
        List<List<Double>> res = new ArrayList<List<Double>>();
        for(int i = 0; i < count; i++) {
            res.add(new ArrayList<Double>());
            for(int j = 0; j < prevCount; j++) {
                res.get(i).add(Math.random() * Math.sqrt(1.0 / prevCount) * 2 - Math.sqrt(1.0 / prevCount));
                //System.out.println(prevCount);
            }
            //System.out.println(res.get(i));
        }
        return res;
    }

    private List<Double> setupBiases(int prevCount) {
        List<Double> res = new ArrayList<Double>();
        for(int i = 0; i < prevCount; i++) {
            res.add(0.0);
        }
        return res;
    }

    public static void write(String file, Network network) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
            out.writeObject(network);
            out.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public static Network read(String file) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
            Network network = (Network) in.readObject();
            in.close();
            return network;
        } catch(IOException e) {
            e.printStackTrace();
        } catch(ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
}
