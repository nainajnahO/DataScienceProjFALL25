import { contextBridge, ipcRenderer } from 'electron'

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld('ipcRenderer', {
    on(channel: string, listener: (event: any, ...args: any[]) => void) {
        ipcRenderer.on(channel, (event, ...args) => listener(event, ...args))
        return this
    },
    off(channel: string, ...args: any[]) {
        const [listener] = args
        if (listener) {
            ipcRenderer.removeListener(channel, listener)
        } else {
            ipcRenderer.removeAllListeners(channel)
        }
        return this
    },
    send(channel: string, ...args: any[]) {
        ipcRenderer.send(channel, ...args)
    },
    invoke(channel: string, ...args: any[]) {
        return ipcRenderer.invoke(channel, ...args)
    },
})
