import { contextBridge, ipcRenderer } from 'electron';
// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', {
    on(channel, listener) {
        ipcRenderer.on(channel, (event, ...args) => listener(event, ...args));
        return this;
    },
    off(channel, ...args) {
        const [listener] = args;
        if (listener) {
            ipcRenderer.removeListener(channel, listener);
        }
        else {
            ipcRenderer.removeAllListeners(channel);
        }
        return this;
    },
    send(channel, ...args) {
        ipcRenderer.send(channel, ...args);
    },
    invoke(channel, ...args) {
        return ipcRenderer.invoke(channel, ...args);
    },
});
