import { useResponseEditor } from "../context/responseEditorContext";
import { useSelector, useDispatch } from "react-redux";
import { setResponseCode, setResponseHighlightMap } from "../store/editorSlice";
import { useEffect, useState } from "react";

export function useResponseData(){

    const dispatch = useDispatch()

    const { responseEditorRef, responseMonacoRef, decorateResponseEditor } = useResponseEditor();
    const response = useSelector((state) => state.editor.response);
    const regenerateResponse = useSelector((state) => state.editor.regenerateResponse);
    const responseDecorations = useSelector((state) => state.editor.responseDecorations);
    const responseHighlightMap = useSelector((state) => state.editor.responseHighlightMap);
    const [tempMap, setTempMap] = useState([]);

    useEffect(() => {

        const lines = response.map(item => item.line);
        const colours = response.map(item => item.colour);

        setTempMap(lines)
        dispatch(setResponseCode(lines.join("\n")));
        dispatch(setResponseHighlightMap(colours));
        

    }, [response, dispatch]);

    useEffect(() => {

        const lines = regenerateResponse.map(item => item.line);
        const colours = regenerateResponse.map(item => item.colour);

        dispatch(setResponseCode(lines.join("\n")));
        dispatch(setResponseHighlightMap(colours))


    }, [regenerateResponse, dispatch]);

    useEffect(() => {
        const editor = responseEditorRef.current;
        const monaco = responseMonacoRef.current;

        if (!editor || !monaco) return;

        const disposable = editor.onDidChangeModelContent(() => {
            if (responseHighlightMap.length) {
                decorateResponseEditor(responseHighlightMap);
            }
        });

        return () => disposable.dispose();
    }, [responseHighlightMap]);

}