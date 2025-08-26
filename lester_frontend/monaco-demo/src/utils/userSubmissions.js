import { useDispatch, useSelector } from "react-redux";
import { callBackend } from "./api";
import {setLoading, setTestResponse, setRegenerateResponse, setResponse} from "../store/editorSlice";
import { useResponseData } from "../hooks/useResponseData";

export function useSubmissionActions(){

    const  dispatch = useDispatch()
    const responseHooks = useResponseData()
    
    const regenerateSelection = useSelector(state => state.editor.regenerateSelection)
    const testSelection = useSelector(state => state.editor.testSelection)
    const inputCode = useSelector(state => state.editor.inputCode)
    const highlightMap = useSelector(state => state.editor.highlightMap)
    const inputs = useSelector(state => state.editor.inputs)
    const loading = useSelector(state => state.editor.loading)

    const regenerateCodeSubmit = async () => {

        if (regenerateSelection==="no_selection") {alert("No regeneration option selected!"); return}
        dispatch(setLoading(true))

        try{
            const result = await callBackend({
                url : "http://127.0.0.1:8000/regenerate_stage/" + regenerateSelection
            })
            dispatch(setRegenerateResponse(result))
        } catch (error) {
            console.log("Backend call failed: ", error)
        } finally {
            dispatch(setLoading(false))
        }
    };

    const testCodeSubmit = async () => {

        if (testSelection === "no_selection") {alert("No test selected!"); return}
        console.log("cheese");
        dispatch(setLoading(true));

        try {
            const result = await callBackend({
                url: "http://127.0.0.1:8000/test_stage/" + testSelection
            });
            dispatch(setTestResponse(result))

        } catch (error) {
            console.error("Backend call failed:", error);
        } finally {
            dispatch(setLoading(false));
        }   
    };

    const submitCode = async() => {
        
        if (loading) return;
        dispatch(setLoading(true));

        try {
        const result = await callBackend({
            url: 'http://127.0.0.1:8000/run',
            payload: { inputCode, highlightMap, manualInputs: inputs },
        });
        dispatch(setResponse(result));

        } catch (error) {
        console.error("Backend call failed:", error);
        } finally {
        dispatch(setLoading(false));
        }          
    };

  return { regenerateCodeSubmit, testCodeSubmit, submitCode };

}

