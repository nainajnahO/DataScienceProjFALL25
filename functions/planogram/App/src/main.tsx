import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter, Routes, Route } from 'react-router-dom'
import App from './App.tsx'
import MachinesPage from './routes'
import PlanogramPage from './routes/machines/[id]/page.tsx'
import LayoutEditorPage from './routes/machines/[id]/layout/page.tsx'
import NotFound from './routes/machines/[id]/not-found.tsx'
import NewMachineEditorPage from './routes/machines/new/page.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<NewMachineEditorPage />} />
          <Route path="machines" element={<MachinesPage />} />
          {/* <Route path="machines/new" element={<NewMachineEditorPage />} /> */}
          <Route path="machines/:id" element={<PlanogramPage />} />
          <Route path="machines/:id/layout" element={<LayoutEditorPage />} />
          <Route path="machines/:id/not-found" element={<NotFound />} />
        </Route>
      </Routes>
    </HashRouter>
  </React.StrictMode>,
)
