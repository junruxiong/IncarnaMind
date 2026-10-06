import { useEffect, useState } from "react";

const SlashMenu = (props: any, ref: any) => {
  const { items } = props;

  const [isSelected, setIsSelected] = useState(0);

  useEffect(() => {
    document.addEventListener("keydown", onKeyDown);
  }, [isSelected]);

  const onKeyDown = (event: any) => {
    if (!props.items) {
      return;
    }
    let indexs = isSelected;
    switch (event.key) {
      case "ArrowUp":
        indexs = (indexs + props.items.length - 1) % props.items.length;
        setIsSelected(indexs);
        break;
      case "ArrowDown":
        indexs = (indexs + 1) % props.items.length;
        break;
      // case "Enter":
      //   indexs = (indexs + 1) % props.items.length;
      //   const item = props.items[indexs];
      //   if (item) {
      //     props.command(item);
      //   }
      //   break;
    }
    setIsSelected(indexs);
  };

  return (
    <div className="flex flex-col bg-white w-[180px] max-h-[240px] overflow-auto rounded-[9px] p-2 shadow-custom_focus outline outline-1	outline-gray-200">
      {items?.map((item: any, index: number) => {
        return (
          <button
            key={index}
            className="text-left text-custom p-1 rounded-[6px]"
            style={{
              backgroundColor:
                isSelected === index ? "rgb(46 50 56 / 13%)" : "",
            }}
            onKeyDown={() => console.log("Enter Key")}
            onClick={() => {
              props.command(item);
            }}
          >
            <div className="" onKeyDown={() => console.log("Enter Key 2")}>
              {item.icon}
              <span>{item.title}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
};

export default SlashMenu;
