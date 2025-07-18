function Loading({loading}){
    return(
    <>
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="flex flex-col items-center">
            <div className="w-10 h-10 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
            <span className="mt-2 text-white">Loading...</span>
          </div>
        </div>
      )}
    </>
    )
}

export default Loading