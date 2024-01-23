import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;

public class Runner {
    public static List<Double> getPixels(String file) {
        try {
            BufferedImage image = ImageIO.read(new File(file));
            ArrayList<Double> list = new ArrayList<>();
            if(image.getWidth() != 28 || image.getHeight() != 28) {
                throw new RuntimeException("Not 28x28");
            }
            for(int i = 0; i < 28; i++) {
                for(int j = 0; j < 28; j++) {
                    Color c = new Color(image.getRGB(j, i));
                    list.add(Math.max(c.getRed(), Math.max(c.getBlue(), c.getGreen())) / 255.0);
                }
            }
            return list;
        } catch(IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("archive/mnist_test.csv"));
        String line = br.readLine();
        line = br.readLine();
        List<List<Integer>> testpixels = new ArrayList<>();
        while(line != null) {
            String[] tokens = line.split(",");
            List<Integer> pixels = new ArrayList<>();
            pixels.add(Integer.parseInt(tokens[0]));
            for(int i = 1; i < tokens.length; i++) {
                pixels.add(Integer.parseInt(tokens[i]));
            }
            testpixels.add(pixels);
            line = br.readLine();
        }
        br.close();
        Trainer trainer = new Trainer(Network.read("trained"));
        trainer.tester(testpixels);
    }
}
