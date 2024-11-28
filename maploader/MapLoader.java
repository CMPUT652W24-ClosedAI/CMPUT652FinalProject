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
    private JFrame frame;
    private PhysicalGameStatePanel panel;

    public MapLoader() {
        // Initialize UnitTypeTable with default configuration
        unitTypeTable = new UnitTypeTable(UnitTypeTable.VERSION_ORIGINAL, UnitTypeTable.MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH);

        // Initialize the JFrame and panel
        frame = new JFrame("Game State Viewer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(512, 512);
        frame.setLayout(new BorderLayout());
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
        // Remove the old panel if it exists
        if (panel != null) {
            frame.remove(panel);
        }

        // Create a new PhysicalGameStatePanel to render the updated game state
        panel = new PhysicalGameStatePanel(gameState);
        panel.setPreferredSize(new Dimension(512, 512));

        // Add the panel to the frame
        frame.add(panel, BorderLayout.CENTER);
        frame.revalidate();
        frame.repaint();

        // Make the window visible (if not already)
        if (!frame.isVisible()) {
            frame.setVisible(true);
        }
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

                // Render the updated game state in the existing window
                mapLoader.renderGameState(gameState);
            } else {
                System.err.println("Failed to load map: " + filePath);
            }
        }
    }
}