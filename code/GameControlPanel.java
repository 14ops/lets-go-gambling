import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.util.HashMap;
import java.util.Map;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Applied Probability and Automation Framework for High-RTP Games
 * Java GUI Control Panel
 * 
 * This control panel allows users to configure and control the betting bot
 * with various strategies and parameters.
 */
public class GameControlPanel extends JFrame {
    
    // UI Components
    private JSpinner boardSizeSpinner;
    private JSpinner mineCountSpinner;
    private JSpinner betAmountSpinner;
    private JComboBox<String> strategyComboBox;
    private JButton startButton;
    private JButton stopButton;
    private JTextArea statusArea;
    private JLabel bankrollLabel;
    private JLabel winRateLabel;
    private JLabel roundsLabel;
    
    // Configuration
    private boolean isRunning = false;
    private Process pythonProcess;
    
    public GameControlPanel() {
        initializeUI();
        setupEventHandlers();
    }
    
    private void initializeUI() {
        setTitle("Applied Probability and Automation Framework - Control Panel");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        // Main panel
        JPanel mainPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        
        // Configuration Panel
        JPanel configPanel = createConfigurationPanel();
        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 2; gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(configPanel, gbc);
        
        // Control Panel
        JPanel controlPanel = createControlPanel();
        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 2; gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(controlPanel, gbc);
        
        // Status Panel
        JPanel statusPanel = createStatusPanel();
        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 2; gbc.fill = GridBagConstraints.BOTH; gbc.weightx = 1.0; gbc.weighty = 1.0;
        mainPanel.add(statusPanel, gbc);
        
        add(mainPanel, BorderLayout.CENTER);
        
        // Set window properties
        setSize(600, 500);
        setLocationRelativeTo(null);
        setResizable(true);
    }
    
    private JPanel createConfigurationPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(new TitledBorder("Game Configuration"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(3, 3, 3, 3);
        
        // Board Size
        gbc.gridx = 0; gbc.gridy = 0; gbc.anchor = GridBagConstraints.WEST;
        panel.add(new JLabel("Board Size:"), gbc);
        boardSizeSpinner = new JSpinner(new SpinnerNumberModel(5, 3, 10, 1));
        gbc.gridx = 1; gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(boardSizeSpinner, gbc);
        
        // Mine Count
        gbc.gridx = 0; gbc.gridy = 1; gbc.fill = GridBagConstraints.NONE;
        panel.add(new JLabel("Number of Mines:"), gbc);
        mineCountSpinner = new JSpinner(new SpinnerNumberModel(3, 1, 15, 1));
        gbc.gridx = 1; gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(mineCountSpinner, gbc);
        
        // Bet Amount
        gbc.gridx = 0; gbc.gridy = 2; gbc.fill = GridBagConstraints.NONE;
        panel.add(new JLabel("Bet Amount ($):"), gbc);
        betAmountSpinner = new JSpinner(new SpinnerNumberModel(1.0, 0.1, 1000.0, 0.1));
        gbc.gridx = 1; gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(betAmountSpinner, gbc);
        
        // Strategy Selection
        gbc.gridx = 0; gbc.gridy = 3; gbc.fill = GridBagConstraints.NONE;
        panel.add(new JLabel("Strategy:"), gbc);
        String[] strategies = {"Takeshi (Aggressive)", "Lelouch (Calculated)", "Kazuya (Conservative)", "Senku (Analytical)"};
        strategyComboBox = new JComboBox<>(strategies);
        gbc.gridx = 1; gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(strategyComboBox, gbc);
        
        return panel;
    }
    
    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new FlowLayout());
        panel.setBorder(new TitledBorder("Bot Control"));
        
        startButton = new JButton("Start Bot");
        startButton.setBackground(new Color(76, 175, 80));
        startButton.setForeground(Color.WHITE);
        startButton.setPreferredSize(new Dimension(120, 35));
        
        stopButton = new JButton("Stop Bot");
        stopButton.setBackground(new Color(244, 67, 54));
        stopButton.setForeground(Color.WHITE);
        stopButton.setPreferredSize(new Dimension(120, 35));
        stopButton.setEnabled(false);
        
        panel.add(startButton);
        panel.add(stopButton);
        
        return panel;
    }
    
    private JPanel createStatusPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(new TitledBorder("Status & Results"));
        
        // Stats panel
        JPanel statsPanel = new JPanel(new GridLayout(1, 3));
        bankrollLabel = new JLabel("Bankroll: $0.00");
        winRateLabel = new JLabel("Win Rate: 0%");
        roundsLabel = new JLabel("Rounds: 0");
        
        bankrollLabel.setHorizontalAlignment(SwingConstants.CENTER);
        winRateLabel.setHorizontalAlignment(SwingConstants.CENTER);
        roundsLabel.setHorizontalAlignment(SwingConstants.CENTER);
        
        statsPanel.add(bankrollLabel);
        statsPanel.add(winRateLabel);
        statsPanel.add(roundsLabel);
        
        panel.add(statsPanel, BorderLayout.NORTH);
        
        // Status text area
        statusArea = new JTextArea();
        statusArea.setEditable(false);
        statusArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        statusArea.setText("Ready to start. Configure parameters and click 'Start Bot'.\n");
        
        JScrollPane scrollPane = new JScrollPane(statusArea);
        scrollPane.setPreferredSize(new Dimension(0, 200));
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    private void setupEventHandlers() {
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                startBot();
            }
        });
        
        stopButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                stopBot();
            }
        });
    }
    
    private void startBot() {
        if (isRunning) return;
        
        try {
            // Create configuration
            Map<String, Object> config = new HashMap<>();
            config.put("board_size", boardSizeSpinner.getValue());
            config.put("mine_count", mineCountSpinner.getValue());
            config.put("bet_amount", betAmountSpinner.getValue());
            config.put("strategy", strategyComboBox.getSelectedItem().toString());
            config.put("mode", "simulation"); // Default to simulation mode
            
            // Write configuration to JSON file
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter writer = new FileWriter("../python-backend/config.json")) {
                gson.toJson(config, writer);
            }
            
            // Update UI
            isRunning = true;
            startButton.setEnabled(false);
            stopButton.setEnabled(true);
            statusArea.append("Bot started with configuration:\n");
            statusArea.append("Board Size: " + config.get("board_size") + "\n");
            statusArea.append("Mines: " + config.get("mine_count") + "\n");
            statusArea.append("Bet Amount: $" + config.get("bet_amount") + "\n");
            statusArea.append("Strategy: " + config.get("strategy") + "\n");
            statusArea.append("Running simulation...\n\n");
            statusArea.setCaretPosition(statusArea.getDocument().getLength());
            
            // Start Python backend (simulation mode)
            startPythonBackend();
            
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Error starting bot: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            resetUI();
        }
    }
    
    private void stopBot() {
        if (!isRunning) return;
        
        try {
            if (pythonProcess != null && pythonProcess.isAlive()) {
                pythonProcess.destroyForcibly();
            }
            
            statusArea.append("Bot stopped by user.\n\n");
            statusArea.setCaretPosition(statusArea.getDocument().getLength());
            
        } catch (Exception ex) {
            statusArea.append("Error stopping bot: " + ex.getMessage() + "\n");
        } finally {
            resetUI();
        }
    }
    
    private void startPythonBackend() {
        try {
            ProcessBuilder pb = new ProcessBuilder("python3", "src/main.py");
            pb.directory(new File("../python-backend"));
            pb.redirectErrorStream(true);
            
            pythonProcess = pb.start();
            
            // Start a thread to read Python output
            new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null && isRunning) {
                        final String output = line;
                        SwingUtilities.invokeLater(() -> {
                            statusArea.append(output + "\n");
                            statusArea.setCaretPosition(statusArea.getDocument().getLength());
                            
                            // Parse output for statistics updates
                            updateStatsFromOutput(output);
                        });
                    }
                } catch (IOException e) {
                    if (isRunning) {
                        SwingUtilities.invokeLater(() -> {
                            statusArea.append("Error reading Python output: " + e.getMessage() + "\n");
                        });
                    }
                } finally {
                    SwingUtilities.invokeLater(() -> resetUI());
                }
            }).start();
            
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this, "Error starting Python backend: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            resetUI();
        }
    }
    
    private void updateStatsFromOutput(String output) {
        // Parse Python output for statistics
        if (output.contains("Bankroll:")) {
            String bankroll = output.substring(output.indexOf("Bankroll:") + 9).trim();
            bankrollLabel.setText("Bankroll: " + bankroll);
        }
        if (output.contains("Win Rate:")) {
            String winRate = output.substring(output.indexOf("Win Rate:") + 9).trim();
            winRateLabel.setText("Win Rate: " + winRate);
        }
        if (output.contains("Rounds:")) {
            String rounds = output.substring(output.indexOf("Rounds:") + 7).trim();
            roundsLabel.setText("Rounds: " + rounds);
        }
    }
    
    private void resetUI() {
        isRunning = false;
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeel());
            } catch (Exception e) {
                // Use default look and feel
            }
            
            new GameControlPanel().setVisible(true);
        });
    }
}

