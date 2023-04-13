import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JLabel;
import java.awt.Dimension;
import java.awt.Color;
import javax.swing.ImageIcon;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.SourceDataLine;

public class VideoPlayer {
    public static void main(String[] args) {
        File videoFile = new File("./InputVideo.rgb"); // name of the RGB video file
        File audioFile = new File("./InputAudio.wav");
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video
        int numFrames = 8682; // number of frames in the video

        // read the audio file and set up the audio output
        try {
            AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
            AudioFormat audioFormat = audioStream.getFormat();
            DataLine.Info info = new DataLine.Info(SourceDataLine.class, audioFormat);
            SourceDataLine audioLine = (SourceDataLine) AudioSystem.getLine(info);
            audioLine.open(audioFormat);
            audioLine.start();

            // read the video file and display each frame
            RandomAccessFile raf = new RandomAccessFile(videoFile, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            ByteBuffer audioBuffer = ByteBuffer.allocate(audioFormat.getFrameSize() * audioFormat.getFrameRate());

            // create the JFrame and JLabel to display the video
            JFrame frame = new JFrame("Video Display");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(new Dimension(width, height));
            frame.setVisible(true);
            JLabel label = new JLabel();
            label.setPreferredSize(new Dimension(width, height));
            frame.add(label);

            for (int i = 0; i < numFrames; i++) {
                buffer.clear();
                channel.read(buffer);
                buffer.rewind();
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        image.setRGB(x, y, rgb);
                    }
                }
                label.setIcon(new ImageIcon(image));
                frame.validate();
                frame.repaint();
                try {
                    Thread.sleep(1000 / fps);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                // play the audio for this frame
                int numBytesRead = audioStream.read(audioBuffer.array());
                if (numBytesRead > 0) {
                    audioLine.write(audioBuffer.array(), 0, numBytesRead);
                }
            }

            channel.close();
            raf.close();
            audioStream.close();
            audioLine.drain();
            audioLine.stop();
            audioLine.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}