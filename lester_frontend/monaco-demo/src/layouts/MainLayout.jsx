function MainLayout({children}){
    return(
      // <div className="grid grid-rows-7 grid-cols-2 h-full w-full max-h-100 col-span-5 flex flex-col gap-4 overflow-y-auto"> 
      <div className="col-start-2 col-span-full row-start-2 row-span-full grid grid-rows-5 bg-neutral-900 border border-neutral-600"> 
        {/* col-span-1 flex flex-col gap-4 overflow-y-auto p-2 bg-neutral-900 border border-neutral-600 */}
        {children}
          
      </div>
    )
}

export default MainLayout