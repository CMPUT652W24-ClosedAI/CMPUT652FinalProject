import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PhysicalGameState;
import rts.units.UnitTypeTable;

import java.util.Scanner;
import javax.swing.*;
import java.awt.*;
import java.io.File;

public class MapLoader {

    private UnitTypeTable unitTypeTable;

    public MapLoader() {
        // Initialize UnitTypeTable with default configuration
        unitTypeTable = new UnitTypeTable(UnitTypeTable.VERSION_ORIGINAL, UnitTypeTable.MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH);
    }

    public GameState loadMap(String filePath) {
        try {
            // Load the PhysicalGameState from the XML file
            PhysicalGameState pgs = PhysicalGameState.load(filePath, unitTypeTable);

            // Create a GameState from the loaded PhysicalGameState
            return new GameState(pgs, unitTypeTable);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void renderGameState(GameState gameState) {
        // Create a JFrame for displaying the game state
        JFrame frame = new JFrame("Game State Viewer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create a PhysicalGameStatePanel to render the game state
        PhysicalGameStatePanel panel = new PhysicalGameStatePanel(gameState);
        panel.setPreferredSize(new Dimension(512, 512));

        // Add the panel to the frame and pack it
        frame.add(panel);
        frame.pack();

        // Center the window on the screen
        frame.setLocationRelativeTo(null);

        // Make the window visible
        frame.setVisible(true);
    }



    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        MapLoader mapLoader = new MapLoader();

        while (true) {
            System.out.println("Enter the map file path (or 'exit' to quit):");
            String filePath = scanner.nextLine();

            if (filePath.equalsIgnoreCase("exit")) {
                System.out.println("Exiting MapLoader.");
                break;
            }

            GameState gameState = mapLoader.loadMap(filePath);
            if (gameState != null) {
                System.out.println("Map loaded successfully!");
                System.out.println("GameState information:");
                System.out.println(gameState.getPhysicalGameState());
                // Optionally visualize the map here
            } else {
                System.err.println("Failed to load map: " + filePath);
            }
        }
    }
    }

    // public static void main(String[] args) {
    //     if (args.length != 1) {
    //         System.err.println("Usage: java -cp \"lib/*:out\" gui.frontend.MapLoader <map-file-path>");
    //         System.exit(1);
    //     }

    //     String filePath = args[0];
    //     File file = new File(filePath);

    //     if (!file.exists() || !file.isFile()) {
    //         System.err.println("Error: The specified file does not exist or is not a valid file: " + filePath);
    //         System.exit(1);
    //     }

    //     MapLoader mapLoader = new MapLoader();

    //     // Load the map
    //     GameState gameState = mapLoader.loadMap(filePath);

    //     if (gameState != null) {
    //         System.out.println("Map loaded successfully!");
    //         System.out.println("GameState information:");
    //         System.out.println(gameState.getPhysicalGameState());

    //         // Render the game state in a window
    //         mapLoader.renderGameState(gameState);
    //     } else {
    //         System.err.println("Failed to load map.");
    //     }
    // }
}
