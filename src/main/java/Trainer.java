

import java.util.*;
import java.io.*;

public class Trainer {
    private Network network;

    public Trainer(Network network) {
        this.network = network;
    }

    public Network getNetwork() {
        return network;
    }

    public void gradient(List<List<Integer>> pixels, double lr) {
        int n = pixels.size();
        List<Layer> layers = network.getLayers();
        List<List<Double>> dw0 = new ArrayList<List<Double>>(); // jk
        List<Double> b0 = new ArrayList<Double>(); // j
        List<List<Double>> dw1 = new ArrayList<List<Double>>(); // jk, calc1 calc2
        List<Double> b1 = new ArrayList<Double>(); // j, calc2
        List<List<Double>> dw2 = new ArrayList<List<Double>>(); // jk, calc1 calc2
        List<Double> b2 = new ArrayList<Double>(); // j, calc2
        List<List<Double>> weights0 = layers.get(layers.size() - 1).getWeights();
        List<Double> biases0 = layers.get(layers.size() - 1).getBiases();
        List<List<Double>> weights1 = layers.get(layers.size() - 2).getWeights();
        List<Double> biases1 = layers.get(layers.size() - 2).getBiases();
        List<List<Double>> weights2 = new ArrayList<List<Double>>();
        List<Double> biases2 = new ArrayList<Double>();
        if (layers.size() == 4) {
            weights2 = layers.get(1).getWeights();
            biases2 = layers.get(1).getBiases();
        }
        for (int i = 0; i < pixels.size(); i++) {
            network.getVal(pixels.get(i).subList(1, pixels.get(i).size()));
            List<Double> calculations0 = layers.get(layers.size() - 1).calculate(null, false);
            List<Double> calculations1 = layers.get(layers.size() - 2).calculate(null, false);
            List<Double> calculations2 = layers.get(layers.size() - 3).calculate(null, false); // hard code input layer
            List<Double> calculations3 = new ArrayList<Double>();
            if (layers.size() == 4) {
                calculations3 = layers.get(0).calculate(null, false);
            }
            List<Double> dcdathing = new ArrayList<Double>();
            List<List<Double>> nextdcda = new ArrayList<List<Double>>();
            for (int j = 0; j < layers.get(layers.size() - 1).getCount(); j++) {
                double dcda = 0.0;
                if (j == pixels.get(i).get(0)) {
                    dcda = 2 * (calculations0.get(j) - 1);
                } else {
                    dcda = 2 * calculations0.get(j);
                }
                dcdathing.add(dcda);
            }
            for (int j = 0; j < layers.get(layers.size() - 1).getCount(); j++) {
                if (dw0.size() <= j) {
                    dw0.add(new ArrayList<Double>());
                }
                double jsigder = calculations0.get(j) * (1 - calculations0.get(j));
                if (b0.size() <= j) {
                    b0.add(0.0);
                }
                //System.out.println(jsigder + " " + dcda);
                double jsigderdcda = jsigder * dcdathing.get(j);
                b0.set(j, b0.get(j) + jsigderdcda);
                nextdcda.add(new ArrayList<Double>());
                for (int k = 0; k < layers.get(layers.size() - 2).getCount(); k++) {
                    if (dw0.get(j).size() <= k) {
                        dw0.get(j).add(0.0);
                    }
                    dw0.get(j).set(k, dw0.get(j).get(k) + calculations1.get(k) * jsigderdcda);
                    nextdcda.get(j).add(weights0.get(j).get(k) * jsigderdcda);
                }
            }
            // second layer
            dcdathing = new ArrayList<Double>();
            for (int k = 0; k < calculations1.size(); k++) {
                double dcda = 0.0;
                for (int j = 0; j < calculations0.size(); j++) {
                    dcda += nextdcda.get(j).get(k);
                }
                dcdathing.add(dcda);
            }
            nextdcda = new ArrayList<List<Double>>();
            for (int j = 0; j < calculations1.size(); j++) {
                if (dw1.size() <= j) {
                    dw1.add(new ArrayList<Double>());
                }
                double jsigder = calculations1.get(j) * (1 - calculations1.get(j));
                if (b1.size() <= j) {
                    b1.add(0.0);
                }
                double jsigderdcda = jsigder * dcdathing.get(j);
                b1.set(j, jsigderdcda);
                nextdcda.add(new ArrayList<Double>());
                for (int k = 0; k < calculations2.size(); k++) {
                    if (dw1.get(j).size() <= k) {
                        dw1.get(j).add(0.0);
                    }
                    dw1.get(j).set(k, dw1.get(j).get(k) + calculations2.get(k) * jsigderdcda);
                    nextdcda.get(j).add(weights1.get(j).get(k) * jsigderdcda);
                }
            }
            dcdathing = new ArrayList<Double>();
            for (int k = 0; k < calculations2.size(); k++) {
                double dcda = 0.0;
                for (int j = 0; j < calculations1.size(); j++) {
                    dcda += nextdcda.get(j).get(k);
                }
                dcdathing.add(dcda);
            }
            nextdcda = new ArrayList<List<Double>>();
            if (layers.size() == 4) {
                for (int j = 0; j < calculations2.size(); j++) {
                    if (dw2.size() <= j) {
                        dw2.add(new ArrayList<Double>());
                    }
                    double jsigder = calculations2.get(j) * (1 - calculations2.get(j));
                    if (b2.size() <= j) {
                        b2.add(0.0);
                    }
                    double jsigderdcda = jsigder * dcdathing.get(j);
                    b2.set(j, jsigderdcda);
                    nextdcda.add(new ArrayList<Double>());
                    for (int k = 0; k < calculations3.size(); k++) {
                        if (dw2.get(j).size() <= k) {
                            dw2.get(j).add(0.0);
                        }
                        dw2.get(j).set(k, dw2.get(j).get(k) + calculations3.get(k) * jsigderdcda);
                        nextdcda.get(j).add(weights2.get(j).get(k) * jsigderdcda);
                    }
                }
            }
        }
        for (int i = 0; i < dw0.size(); i++) {
            b0.set(i, b0.get(i) / n);
            for (int j = 0; j < dw0.get(i).size(); j++) {
                dw0.get(i).set(j, dw0.get(i).get(j) / n);
            }
        }
        for (int i = 0; i < dw1.size(); i++) {
            b1.set(i, b1.get(i) / n);
            for (int j = 0; j < dw1.get(i).size(); j++) {
                dw1.get(i).set(j, dw1.get(i).get(j) / n);
            }
        }
        if (layers.size() == 4) {
            for (int i = 0; i < dw2.size(); i++) {
                b2.set(i, b2.get(i) / n);
                for (int j = 0; j < dw2.get(i).size(); j++) {
                    dw2.get(i).set(j, dw2.get(i).get(j) / n);
                }
            }
        }

        //System.out.println(biases0);
        //System.out.println(dw0);
        for (int i = 0; i < weights0.size(); i++) {
            biases0.set(i, biases0.get(i) - lr * b0.get(i)); //accuracy bad but cost good
            for (int j = 0; j < weights0.get(i).size(); j++) {
                weights0.get(i).set(j, weights0.get(i).get(j) - lr * dw0.get(i).get(j)); //good
            }
        }
        //System.out.println(biases0);
        //System.out.println(dw1);
        for (int i = 0; i < weights1.size(); i++) {
            biases1.set(i, biases1.get(i) - lr * b1.get(i));  //literally nothing it's random
            for (int j = 0; j < weights1.get(i).size(); j++) {
                weights1.get(i).set(j, weights1.get(i).get(j) - lr * dw1.get(i).get(j)); //good
            }
        }
        if (layers.size() == 4) {
            //System.out.println(b2);
            //System.out.println(dw2);
            //System.out.println("\n\n" + dw2.size());
            for (int i = 0; i < weights2.size(); i++) {
                biases2.set(i, biases2.get(i) - lr * b2.get(i));
                //System.out.println(dw2.get(i).size());
                for (int j = 0; j < weights2.get(i).size(); j++) {
                    weights2.get(i).set(j, weights2.get(i).get(j) - lr * dw2.get(i).get(j));
                }
            }
        }
    }

    public double tester(List<List<Integer>> pixels) {
        int total = pixels.size();
        int correct = 0;
        double cost = 0.0;
        for (int i = 0; i < pixels.size(); i++) {
            if (network.getVal(pixels.get(i).subList(1, pixels.get(i).size())) == pixels.get(i).get(0)) {
                correct++;
            }
            cost += network.getCost(pixels.get(i).get(0));
        }
        System.out.println("Results: " + correct + "/" + total + " = " + 1.0 * correct / total);
        System.out.println("Average Cost: " + cost / pixels.size());
        return 1.0 * correct / total;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("archive/mnist_train.csv"));
        String line = br.readLine();
        line = br.readLine();
        List<List<Integer>> pixelss = new ArrayList<List<Integer>>();
        while (line != null) {
            String[] tokens = line.split(",");
            List<Integer> pixels = new ArrayList<Integer>();
            pixels.add(Integer.parseInt(tokens[0]));
            for (int i = 1; i < tokens.length; i++) {
                pixels.add(Integer.parseInt(tokens[i]));
            }
            pixelss.add(pixels);
            line = br.readLine();
        }
        br.close();
        br = new BufferedReader(new FileReader("archive/mnist_test.csv"));
        line = br.readLine();
        line = br.readLine();
        List<List<Integer>> tpixelss = new ArrayList<List<Integer>>();
        while (line != null) {
            String[] tokens = line.split(",");
            List<Integer> pixels = new ArrayList<Integer>();
            pixels.add(Integer.parseInt(tokens[0]));
            for (int i = 1; i < tokens.length; i++) {
                pixels.add(Integer.parseInt(tokens[i]));
            }
            tpixelss.add(pixels);
            line = br.readLine();
        }
        br.close();
        System.out.println("\n\n\n?\n");
        //Trainer trainer = new Trainer(new Network(new int[]{pixelss.get(0).size() - 1, 80, 40, 10}));
        Trainer trainer = new Trainer(Network.read("trained"));
        trainer.tester(tpixelss);

        double correct = 0;
        Queue<Double> last = new LinkedList<Double>();
        last.offer(0.0);
        for (int i = 0; i < 1000; i++) {
            Collections.shuffle(pixelss);
            for (int j = 0; j < pixelss.size() / 100; j++) {
                //System.out.print("" + j + ".\t" + "");
                trainer.gradient(pixelss.subList(j * 100, j * 100 + 100), 0.01);
                //trainer.tester(tpixelss, tlabels);
            }
            correct = trainer.tester(tpixelss);
            double avg = 0.0;
            for (double d : last) {
                avg += d;
            }
            avg /= last.size();
            if (correct < avg + 0) {
                break;
            }
            last.offer(correct);
            if (last.size() > 3) {
                last.poll();
            }
        }
        // Network.write("albania5", trainer.getNetwork());
    }
}
