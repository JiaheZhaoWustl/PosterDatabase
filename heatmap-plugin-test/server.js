const express = require('express');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(bodyParser.json());

// Endpoint to receive console output from Figma plugin
app.post('/log', (req, res) => {
  const { message, level = 'info', meta = {} } = req.body;
  const logLine = `${new Date().toISOString()} [${level}] ${message} ${JSON.stringify(meta)}\n`;
  const logPath = path.join(__dirname, 'console.log');
  fs.appendFile(logPath, logLine, (err) => {
    if (err) {
      console.error('Failed to write log:', err);
      return res.status(500).json({ success: false, error: 'Failed to write log' });
    }
    // Placeholder for triggering other backend actions
    // e.g., triggerAction(meta)
    res.json({ success: true });
  });
});

app.get('/', (req, res) => {
  res.send('Figma Plugin Log Server is running.');
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
}); 