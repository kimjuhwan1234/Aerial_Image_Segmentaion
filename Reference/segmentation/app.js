const { spawn } = require('child_process');

const command = spawn('yolo', ['detect', 'train', 'data=data.yaml', 'model=model/yolov8l.pt', 'epochs=10', 'imgsz=736']);

command.stdout.on('data', data => {
  const output = data.toString(); // Convert the buffer to a string
  console.log(`stdout: ${output}`);
});

command.stderr.on('data', data => {
  const output = data.toString(); // Convert the buffer to a string
  console.error(`stderr: ${output}`);
});

command.on('error', error => {
  console.error(`error: ${error.message}`);
});

command.on('close', code => {
  console.log(`child process exited with code ${code}`);
  
  // Check the exit code to determine if training was successful
  if (code === 0) {
    console.log("Training completed successfully. Terminating the main process...");
    process.exit(0); // Terminate the main process (app.js)
  } else {
    console.log("Training process exited with an error.");
    // Handle the error or take appropriate action.
    process.exit(1); // Terminate the main process with a non-zero exit code
  }
});
