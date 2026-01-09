import { app, BrowserWindow, ipcMain, dialog, net } from 'electron';
import { spawn } from 'child_process';
import path from 'path'; // Type error if @types/node not installed, but it usually is in Vite projects.
import { fileURLToPath } from 'url';
import fs from 'fs';
// Necessary for ES modules in Electron if using strict ESM
const __dirname = path.dirname(fileURLToPath(import.meta.url));
// process.env.DIST = path.join(__dirname, '../dist')
// process.env.VITE_PUBLIC = app.isPackaged ? process.env.DIST : path.join(process.env.DIST, '../public')
let win;
let pythonProcess = null;
const PY_PORT = 5001;
const PY_URL = `http://127.0.0.1:${PY_PORT}`;
function startPythonSubprocess() {
    // Determine the correct base path depending on whether app is packaged or in development
    let basePath;
    if (app.isPackaged) {
        // When packaged, go up from the .app bundle to the App directory
        // Path: App/release/mac-arm64/My App.app/Contents/Resources
        // We need to go up to the App directory, then up to planogram
        basePath = path.resolve(process.resourcesPath, '../../../../..');
    }
    else {
        // In development, __dirname is in the electron directory or dist-electron
        basePath = path.resolve(__dirname, '..');
    }
    const pythonScript = path.resolve(basePath, '../new_location_engine/api.py');
    const pythonExecutable = path.resolve(basePath, '../../../.venv/bin/python');
    console.log('App packaged:', app.isPackaged);
    console.log('process.resourcesPath:', process.resourcesPath);
    console.log('__dirname:', __dirname);
    console.log('Base path:', basePath);
    console.log('Python executable:', pythonExecutable);
    console.log('Python script:', pythonScript);
    // Check if files exist before spawning
    if (!fs.existsSync(pythonExecutable)) {
        console.error('Python executable not found at:', pythonExecutable);
        dialog.showErrorBox('Python Not Found', `Python executable not found at:\n${pythonExecutable}\n\nPlease check the paths in main.ts`);
        return;
    }
    if (!fs.existsSync(pythonScript)) {
        console.error('Python script not found at:', pythonScript);
        dialog.showErrorBox('API Script Not Found', `API script not found at:\n${pythonScript}\n\nPlease check the paths in main.ts`);
        return;
    }
    console.log('Starting Python subprocess...');
    // Use the venv python to run the script using -u for unbuffered output
    pythonProcess = spawn(pythonExecutable, ['-u', pythonScript]);
    if (pythonProcess.stdout) {
        pythonProcess.stdout.on('data', (data) => {
            console.log(`[Python]: ${data}`);
        });
    }
    if (pythonProcess.stderr) {
        pythonProcess.stderr.on('data', (data) => {
            console.error(`[Python stderr]: ${data}`);
        });
    }
    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        pythonProcess = null;
    });
}
function createWindow() {
    win = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.mjs'),
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false, // Important: allows preload to work if contextIsolation is on in some environments
        },
    });
    // Test active push message to user  // Test active push message to user 
    if (process.env.NODE_ENV === 'development' || process.env.VITE_DEV_SERVER_URL) {
        win.loadURL('http://localhost:5173');
        // Open DevTools automatically in dev mode
        win.webContents.openDevTools();
    }
    else {
        // win.loadFile('dist/index.html')
        win.loadFile(path.join(__dirname, '../dist/index.html'));
    }
}
app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});
ipcMain.handle('capture-planogram', async (_event, rect) => {
    if (!win)
        return { success: false, error: 'No window' };
    try {
        // 1. Prepare for CDP (Chrome DevTools Protocol)
        if (!win.webContents.debugger.isAttached()) {
            try {
                win.webContents.debugger.attach('1.3');
            }
            catch (err) {
                console.error('Debugger attach failed:', err);
                return { success: false, error: 'Debugger attach failed' };
            }
        }
        // 2. Capture Screenshot via CDP
        // We use Page.captureScreenshot which supports captureBeyondViewport
        const { data } = await win.webContents.debugger.sendCommand('Page.captureScreenshot', {
            format: 'png',
            captureBeyondViewport: true,
            clip: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                scale: 1
            }
        });
        const pngBuffer = Buffer.from(data, 'base64');
        // 3. Show Save Dialog
        const { filePath } = await dialog.showSaveDialog(win, {
            title: 'Save Planogram',
            defaultPath: `planogram-${Date.now()}.png`,
            filters: [{ name: 'PNG Image', extensions: ['png'] }]
        });
        if (filePath) {
            // 4. Write to file
            fs.writeFileSync(filePath, pngBuffer);
            return { success: true };
        }
        else {
            return { success: false, error: 'Cancelled' };
        }
    }
    catch (error) {
        console.error('Capture failed:', error);
        return { success: false, error: error.message };
    }
    finally {
        // Optional: detach if you don't want to leave it open, but generally safe to keep attached or check next time
        if (win.webContents.debugger.isAttached()) {
            win.webContents.debugger.detach();
        }
    }
});
// Helper to wait
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
ipcMain.handle('predict-location', async (_event, data) => {
    const maxRetries = 10; // Wait up to ~10-20 seconds for startup
    let attempt = 0;
    while (attempt < maxRetries) {
        try {
            return await new Promise((resolve, reject) => {
                const request = net.request({
                    method: 'POST',
                    url: `${PY_URL}/predict`,
                });
                request.setHeader('Content-Type', 'application/json');
                request.on('response', (response) => {
                    let body = '';
                    response.on('data', (chunk) => {
                        body += chunk.toString();
                    });
                    response.on('end', () => {
                        try {
                            const result = JSON.parse(body);
                            if (response.statusCode === 200) {
                                resolve(result);
                            }
                            else {
                                reject(new Error(result.error || `Error ${response.statusCode}`));
                            }
                        }
                        catch (e) {
                            reject(e);
                        }
                    });
                });
                request.on('error', (error) => {
                    reject(error);
                });
                request.write(JSON.stringify(data));
                request.end();
            });
        }
        catch (error) {
            if ((error.code === 'ECONNREFUSED' || error.message.includes('ERR_CONNECTION_REFUSED')) && attempt < maxRetries) {
                console.log(`Connection refused, retrying in 2s (Attempt ${attempt + 1}/${maxRetries})...`);
                attempt++;
                await wait(2000);
            }
            else {
                throw error;
            }
        }
    }
});
ipcMain.handle('autofill-machine', async (_event, data) => {
    // data: { slots: [], machineId: string, action: 'fill' | 'optimize' }
    const maxRetries = 10;
    let attempt = 0;
    while (attempt < maxRetries) {
        try {
            return await new Promise((resolve, reject) => {
                const request = net.request({
                    method: 'POST',
                    url: `${PY_URL}/autofill`,
                });
                request.setHeader('Content-Type', 'application/json');
                request.on('response', (response) => {
                    let body = '';
                    response.on('data', (chunk) => {
                        body += chunk.toString();
                    });
                    response.on('end', () => {
                        try {
                            const result = JSON.parse(body);
                            if (response.statusCode === 200) {
                                resolve(result);
                            }
                            else {
                                reject(new Error(result.error || `Error ${response.statusCode}`));
                            }
                        }
                        catch (e) {
                            reject(e);
                        }
                    });
                });
                request.on('error', (error) => {
                    reject(error);
                });
                request.write(JSON.stringify(data));
                request.end();
            });
        }
        catch (error) {
            if ((error.code === 'ECONNREFUSED' || error.message.includes('ERR_CONNECTION_REFUSED')) && attempt < maxRetries) {
                console.log(`Connection refused, retrying in 2s (Attempt ${attempt + 1}/${maxRetries})...`);
                attempt++;
                await wait(2000);
            }
            else {
                throw error;
            }
        }
    }
});
app.whenReady().then(() => {
    startPythonSubprocess();
    createWindow();
});
