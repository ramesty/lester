import Editor from "@monaco-editor/react";
import { useEditor } from "../context/editorContext";
import { useSelector } from "react-redux";

function CodeEditor(){

    const { handleCodeChange, handleEditorDidMount } = useEditor();
    const inputCode = useSelector((state) => state.editor.inputCode)
    const validCode = useSelector((state) => state.editor.validCode)

    return(

        <div className="col-start-1 col-span-1 p-4 vs-bg-dark border border-neutral-600">
          
          <h2 className="p-2 text-center text-lg">Input Code</h2>
          
          <div className="h-full max-h-140">
            <Editor
              defaultLanguage="python"
              value={inputCode}
              onChange= {handleCodeChange}
              onMount={handleEditorDidMount}
              theme="vs-dark"
              options={{ readOnly: false, minimap: { enabled: false } }}
              height="100%"
            />
          </div>
        </div>
    )
}

export default CodeEditor