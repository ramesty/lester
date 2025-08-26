import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { Provider } from "react-redux";
import { store } from "./store/store";
import { EditorProvider } from './context/editorContext.jsx';
import { ResponseEditorProvider } from './context/responseEditorContext.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Provider store={store}>
      <EditorProvider>
        <ResponseEditorProvider>
          <App></App>
        </ResponseEditorProvider>
      </EditorProvider>
    </Provider>
  </StrictMode>,
)
