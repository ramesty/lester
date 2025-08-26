import React, { useRef, createContext, useContext } from "react";
import { setResponseDecorations, setResponseCode } from "../store/editorSlice";
import { useDispatch, useSelector } from "react-redux";

const ResponseEditorContext = createContext(null);

export const ResponseEditorProvider = ({ children }) => {
    
    const dispatch = useDispatch();
    const responseEditorRef = useRef(null);
    const responseMonacoRef = useRef(null);

    const responseDecorations = useSelector((state)=> state.editor.responseDecorations)
    const responseHighlightMap = useSelector((state) => state.editor.responseHighlightMap);
    const responseCode = useSelector((state) => state.editor.responseCode);


    const handleResponseEditorDidMount = (responseEditor, responseMonaco) => {
        responseEditorRef.current = responseEditor;
        responseMonacoRef.current = responseMonaco; 
    };

    const decorateResponseEditor = (responseHMap) => {

        const responseEditor = responseEditorRef.current;
        const responseMonaco = responseMonacoRef.current;

        responseEditor.deltaDecorations(responseDecorations, []);
        dispatch(setResponseDecorations([]));

        const lines = responseCode.split("\n");
        const newDecorations = responseHMap.map((clr, index) => {

            const lineLength = lines[index]?.length || 1;

            return {
                range: new responseMonaco.Range(index + 1, 1, index + 1, lineLength),
                options: {
                    // isWholeLine: true,
                    className: `highlight-${clr}`,
                },
            };
        });

        const decorationIds = responseEditor.deltaDecorations([], newDecorations);
        dispatch(setResponseDecorations(decorationIds));
    };

    const handleResponseEditorCodeChange = (updatedResponseCode) => {
        // const editor = responseEditorRef.current;
        // const monaco = responseMonacoRef.current;
        // decorateResponseEditor(editor, monaco)
    }

    return (
        <ResponseEditorContext.Provider value={{ responseEditorRef, responseMonacoRef, handleResponseEditorDidMount, decorateResponseEditor, handleResponseEditorCodeChange }}>
        {children}
        </ResponseEditorContext.Provider>
    );
};

export const useResponseEditor = () => useContext(ResponseEditorContext);
