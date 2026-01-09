// Mock data for vending machines
import type { AppMachine } from "./types.ts"

export const MOCK_MACHINES: AppMachine[] = [
  {
    id: "2",
    machine_name: "Testing Machine",
    machine_model: "Sielaff 6000",
    machine_key: "TERM-A-001",
    machine_sub_group: "Airport Group",
    last_sale: "2025-10-28",
    n_sales: 1234,
    refillers: ["John Smith", "Sarah Johnson"],
    location: {
      address: "Terminal A, Gate 12",
      city: "Yerevan",
      country: "Armenia",
    },
    slots: [
      {
        position: "A0",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1.5,
        is_discount: false,
        discount: 0,
      },
      {
        position: "A1",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1.5,
        is_discount: false,
        discount: 0,
      },
      {
        position: "A5",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 3,
        is_discount: false,
        discount: 0,
      },
      {
        position: "C5",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1.5,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D4",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1.5,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D1",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1.5,
        is_discount: false,
        discount: 0,
      },
    ],
  },
  {
    id: "1",
    machine_name: "Airport Terminal A",
    machine_model: "Sielaff 6000",
    machine_key: "TERM-A-001",
    machine_sub_group: "Airport Group",
    last_sale: "2025-10-28",
    n_sales: 1234,
    refillers: ["John Smith", "Sarah Johnson"],
    location: {
      address: "Terminal A, Gate 12",
      city: "Yerevan",
      country: "Armenia",
    },
    slots: [
      // Row A
      {
        position: "A0",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 0,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "A1",
        product_name: "Pepsi",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FPepsi%2033cl.png?alt=media&token=c62c7147-a32d-4e90-999f-57fdf3d56b2a",
        price: 26,
        stock_current: 3,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "A2",
        price: 0,
        stock_current: 0,
        stock_max: 10,
        width: 1,
        is_discount: false,
        discount: 0,
        product_name: "",
        image_url: "",
        category: ""
      },
      {
        position: "A3",
        product_name: "Water",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FAqua%20D'or%20Stilla%2050cl.png?alt=media&token=b513c523-6eeb-4eb6-ae39-cf02559c6ddd",
        price: 16,
        stock_current: 8,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "A4",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 6,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "A5",
        price: 0,
        stock_current: 0,
        stock_max: 10,
        width: 1,
        is_discount: false,
        discount: 0,
        product_name: "",
        image_url: "",
        category: ""
      },
      {
        position: "A6",
        product_name: "Lasagnette Bolognese",
        image_url:
          "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Gooh%21%20Lasagnette%20Bolognese.png?Expires=2220457368&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=GEFWhExetVJC8OSC%2BVZUL%2BdMuhwXxPMf2UYQx46zAyX9buJsJ%2Bm3TquTZbb7X%2FCDZY3bLHB9nvGa0cru%2F5CGIQ92PKTbRp45wPeimdUhNsx2Osbnd%2F20uHLLXY9bO0zKmyRIcXFHpsIb2KsrKw6pwWAZUGa4fOn%2BQLt4xTdGOwbeZTELjEDtbJK61BsvauQW1tjzy%2FtPp2UM%2FAorK2lmYe1kv0j4LYpJgk2h0wgcLWjyYQD89Ptme4ZH8WYdaTLr32HLJPcqImsz5B%2BCUzAoeA1Y0PBIQNdJLihx3beZJzrSe0O6c81E2vmnO5NT6wOlLThzrSlRwFNisU73u3yoSw%3D%3D",
        price: 18,
        stock_current: 7,
        stock_max: 10,
        category: "Food",
        width: 3,
        is_discount: false,
        discount: 0,
      },
      // Row B
      {
        position: "B0",
        product_name: "Lasagnette Bolognese",
        image_url:
          "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Gooh%21%20Lasagnette%20Bolognese.png?Expires=2220457368&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=GEFWhExetVJC8OSC%2BVZUL%2BdMuhwXxPMf2UYQx46zAyX9buJsJ%2Bm3TquTZbb7X%2FCDZY3bLHB9nvGa0cru%2F5CGIQ92PKTbRp45wPeimdUhNsx2Osbnd%2F20uHLLXY9bO0zKmyRIcXFHpsIb2KsrKw6pwWAZUGa4fOn%2BQLt4xTdGOwbeZTELjEDtbJK61BsvauQW1tjzy%2FtPp2UM%2FAorK2lmYe1kv0j4LYpJgk2h0wgcLWjyYQD89Ptme4ZH8WYdaTLr32HLJPcqImsz5B%2BCUzAoeA1Y0PBIQNdJLihx3beZJzrSe0O6c81E2vmnO5NT6wOlLThzrSlRwFNisU73u3yoSw%3D%3D",
        price: 18,
        stock_current: 8,
        stock_max: 12,
        category: "Food",
        width: 2,
        is_discount: false,
        discount: 0,
      },
      {
        position: "B2",
        product_name: "Pringles",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FPringles%20Original%2040g.png?alt=media&token=533202e6-3f8b-485c-9ce6-f2f5371c87be",
        price: 29,
        stock_current: 4,
        stock_max: 8,
        category: "Snacks",
        width: 1.5,
        is_discount: false,
        discount: 0,
      },
      
       {
        position: "B5",
        product_name: "Snickers",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
        price: 18,
        stock_current: 5,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "B6",
        product_name: "Snickers",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
        price: 18,
        stock_current: 5,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
     
      {
        position: "B7",
        product_name: "M&Ms",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FM%26M%20100g.png?alt=media&token=24d116a4-8911-4bc1-8c2e-b4a39fb07e6f",
        price: 16,
        stock_current: 8,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },

      // Row C
      
      {
        position: "C1",
        product_name: "Pasta Bolognese",
        image_url:
          "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Gooh%21%20Lasagnette%20Bolognese.png?Expires=2220457368&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=GEFWhExetVJC8OSC%2BVZUL%2BdMuhwXxPMf2UYQx46zAyX9buJsJ%2Bm3TquTZbb7X%2FCDZY3bLHB9nvGa0cru%2F5CGIQ92PKTbRp45wPeimdUhNsx2Osbnd%2F20uHLLXY9bO0zKmyRIcXFHpsIb2KsrKw6pwWAZUGa4fOn%2BQLt4xTdGOwbeZTELjEDtbJK61BsvauQW1tjzy%2FtPp2UM%2FAorK2lmYe1kv0j4LYpJgk2h0wgcLWjyYQD89Ptme4ZH8WYdaTLr32HLJPcqImsz5B%2BCUzAoeA1Y0PBIQNdJLihx3beZJzrSe0O6c81E2vmnO5NT6wOlLThzrSlRwFNisU73u3yoSw%3D%3D",
        price: 29,
        stock_current: 3,
        stock_max: 8,
        category: "Food",
        width: 2,
        is_discount: false,
        discount: 0,
      },
      
      {
        position: "C4",
        product_name: "KitKat",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FKitkat.png?alt=media&token=b655559f-d39f-4032-83df-ae94e8322ab2",
        price: 18,
        stock_current: 4,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "C5",
        product_name: "Snickers",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
        price: 18,
        stock_current: 9,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "C6",
        product_name: "M&Ms",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FM%26M%20100g.png?alt=media&token=24d116a4-8911-4bc1-8c2e-b4a39fb07e6f",
        price: 16,
        stock_current: 6,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
     
      
      // Row D
      {
        position: "D0",
        product_name: "Water",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FAqua%20D'or%20Stilla%2050cl.png?alt=media&token=b513c523-6eeb-4eb6-ae39-cf02559c6ddd",
        price: 16,
        stock_current: 6,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D1",
        product_name: "Sprite",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSprite%2033cl.png?alt=media&token=97ca9c19-295a-40bc-b657-489c0d105818",
        price: 23,
        stock_current: 4,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D2",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D3",
        product_name: "Pepsi",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FPepsi%2033cl.png?alt=media&token=c62c7147-a32d-4e90-999f-57fdf3d56b2a",
        price: 26,
        stock_current: 7,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D4",
        product_name: "Water",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FAqua%20D'or%20Stilla%2050cl.png?alt=media&token=b513c523-6eeb-4eb6-ae39-cf02559c6ddd",
        price: 16,
        stock_current: 8,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D5",
        product_name: "Sprite",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSprite%2033cl.png?alt=media&token=97ca9c19-295a-40bc-b657-489c0d105818",
        price: 23,
        stock_current: 6,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D6",
        product_name: "Coca Cola",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
        price: 26,
        stock_current: 4,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D7",
        product_name: "Pepsi",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FPepsi%2033cl.png?alt=media&token=c62c7147-a32d-4e90-999f-57fdf3d56b2a",
        price: 26,
        stock_current: 5,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "D8",
        product_name: "Water",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FAqua%20D'or%20Stilla%2050cl.png?alt=media&token=b513c523-6eeb-4eb6-ae39-cf02559c6ddd",
        price: 16,
        stock_current: 9,
        stock_max: 10,
        category: "Beverages",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      // Row E
      {
        position: "E0",
        product_name: "KitKat",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FKitkat.png?alt=media&token=b655559f-d39f-4032-83df-ae94e8322ab2",
        price: 18,
        stock_current: 7,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "E1",
        product_name: "Snickers",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
        price: 18,
        stock_current: 5,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "E2",
        product_name: "M&Ms",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FM%26M%20100g.png?alt=media&token=24d116a4-8911-4bc1-8c2e-b4a39fb07e6f",
        price: 16,
        stock_current: 8,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "E3",
        product_name: "Twix",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FTwix%2050g%203st.png?alt=media&token=82fcebe7-fd52-490f-8d5c-2b5fd3b98342",
        price: 18,
        stock_current: 6,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      
      {
        position: "E5",
        product_name: "Pasta Bolognese",
        image_url:
          "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Gooh%21%20Lasagnette%20Bolognese.png?Expires=2220457368&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=GEFWhExetVJC8OSC%2BVZUL%2BdMuhwXxPMf2UYQx46zAyX9buJsJ%2Bm3TquTZbb7X%2FCDZY3bLHB9nvGa0cru%2F5CGIQ92PKTbRp45wPeimdUhNsx2Osbnd%2F20uHLLXY9bO0zKmyRIcXFHpsIb2KsrKw6pwWAZUGa4fOn%2BQLt4xTdGOwbeZTELjEDtbJK61BsvauQW1tjzy%2FtPp2UM%2FAorK2lmYe1kv0j4LYpJgk2h0wgcLWjyYQD89Ptme4ZH8WYdaTLr32HLJPcqImsz5B%2BCUzAoeA1Y0PBIQNdJLihx3beZJzrSe0O6c81E2vmnO5NT6wOlLThzrSlRwFNisU73u3yoSw%3D%3D",
        price: 29,
        stock_current: 5,
        stock_max: 8,
        category: "Food",
        width: 2,
        is_discount: false,
        discount: 0,
      },
      {
        position: "E7",
        product_name: "KitKat",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FKitkat.png?alt=media&token=b655559f-d39f-4032-83df-ae94e8322ab2",
        price: 18,
        stock_current: 6,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "E8",
        product_name: "Snickers",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
        price: 18,
        stock_current: 7,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      // Row F
      
      {
        position: "F1",
        product_name: "M&Ms",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FM%26M%20100g.png?alt=media&token=24d116a4-8911-4bc1-8c2e-b4a39fb07e6f",
        price: 16,
        stock_current: 5,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "F2",
        product_name: "Twix",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FTwix%2050g%203st.png?alt=media&token=82fcebe7-fd52-490f-8d5c-2b5fd3b98342",
        price: 18,
        stock_current: 8,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "F3",
        product_name: "KitKat",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FKitkat.png?alt=media&token=b655559f-d39f-4032-83df-ae94e8322ab2",
        price: 18,
        stock_current: 4,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "F4",
        product_name: "Snickers",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
        price: 18,
        stock_current: 7,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      
      {
        position: "F6",
        product_name: "M&Ms",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FM%26M%20100g.png?alt=media&token=24d116a4-8911-4bc1-8c2e-b4a39fb07e6f",
        price: 16,
        stock_current: 6,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "F7",
        product_name: "Twix",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FTwix%2050g%203st.png?alt=media&token=82fcebe7-fd52-490f-8d5c-2b5fd3b98342",
        price: 18,
        stock_current: 9,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
      {
        position: "F8",
        product_name: "KitKat",
        image_url:
          "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FKitkat.png?alt=media&token=b655559f-d39f-4032-83df-ae94e8322ab2",
        price: 18,
        stock_current: 8,
        stock_max: 12,
        category: "Candy",
        width: 1,
        is_discount: false,
        discount: 0,
      },
    ],
  },
  
]

export const PRODUCT_CATALOG = [
  {
    id: "1",
    name: "Coca Cola",
    price: 26,
    category: "Beverages",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media&token=fba2c011-d91b-4900-a48c-4bd3284028b8",
  },
  {
    id: "2",
    name: "Pepsi",
    price: 26,
    category: "Beverages",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FPepsi%2033cl.png?alt=media&token=c62c7147-a32d-4e90-999f-57fdf3d56b2a",
  },
  {
    id: "3",
    name: "Sprite",
    price: 23,
    category: "Beverages",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSprite%2033cl.png?alt=media&token=97ca9c19-295a-40bc-b657-489c0d105818",
  },
  {
    id: "4",
    name: "Water",
    price: 16,
    category: "Beverages",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FAqua%20D'or%20Stilla%2050cl.png?alt=media&token=b513c523-6eeb-4eb6-ae39-cf02559c6ddd",
  },
  {
    id: "5",
    name: "Snickers",
    price: 18,
    category: "Candy",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FSnickers%2050g.png?alt=media&token=60f600f5-9997-4ae1-99fb-dd65cc45d068",
  },
  {
    id: "6",
    name: "M&Ms",
    price: 16,
    category: "Candy",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FM%26M%20100g.png?alt=media&token=24d116a4-8911-4bc1-8c2e-b4a39fb07e6f",
  },
 
  {
    id: "10",
    name: "Pringles",
    price: 29,
    category: "Snacks",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FPringles%20Original%2040g.png?alt=media&token=533202e6-3f8b-485c-9ce6-f2f5371c87be",
  },
  {
    id: "11",
    name: "Twix",
    price: 18,
    category: "Candy",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FTwix%2050g%203st.png?alt=media&token=82fcebe7-fd52-490f-8d5c-2b5fd3b98342",
  },
  {
    id: "12",
    name: "KitKat",
    price: 18,
    category: "Candy",
    image:
      "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FKitkat.png?alt=media&token=b655559f-d39f-4032-83df-ae94e8322ab2",
  },
  {
    id: "13",
    name: "Cloetta Egg",
    price: 18,
    category: "Candy",
    image:
    "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Cloetta%20Egg.png?Expires=2220457344&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=c5hMiBKqdQj6emXlOwgHIsBvhT17Gyc37j4U1j%2F1K88zOIxORdodJAGGqV%2BW9JGV0QYPJ%2FtWIvei3iK%2Bk7UQJ%2BRczxWMAw7BgO07tv2sJu6zZ8RT1K6yykPN9f1GNGlrqmwUF6RRT9UEpyx8bTy6hGzIu4d3ObZmWGrHPbuCUBOIHhJMm3UP%2BsCACoJeOWZuRBf%2BNDUnHF6rLRebG0erKZkxOi9lJi4KCb2fH0EdmLW3lLqnBja327rIfDL%2FtURiLcExx%2FLVBHMR4DoKtlitohnJ8OhR6C%2Fu9oxb3JU2pqiakuOYxUMxPSEYZnAu1Y6zlm5vhpynF0ELmyK0iV7vJA%3D%3D",
  },
  {
    id: "14",
    name: "Thai Cup Noodles Chicken",
    price: 18,
    category: "Candy",
    image:
    "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Thai%20Cup%20Noodles%20Chicken.png?Expires=2220457344&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=kGM0RXzPB0Z04%2BT4x6B8%2FmVl%2BOF0YP9ub4mwJnMawkKLVpXAJeFfiNW8PunNTbGf5WK4a3wEd6of2bRD8vapTW75N04ywPROcZvMc41N7PShKPTg%2FKTJyNEfqtTbQ%2BPnF%2F%2BaEgWcOYHmLlQf1f1Lnf%2Fe1%2BeTSnJbymKpjb174FwLvRrw4mwctTPrTmRBclQ8SuuMdDQbRBKkknxICDJg%2BY6pLqNBqFmVhI9vUcPyUrwi8%2Fo0oz50mCynlh1rnAAA16QObKfIXr4y7bimaFzLeuw14iXsKI4IuSK7xzsgToOk5CVOlCMtUf8jwtfUlxqa8gra7rgmdXlnIgC1Q4i%2FIA%3D%3D",
  },
  {
    id: "15",
    name: "Risifrutti Hallon",
    price: 18,
    category: "Candy",
    image:
    "https://storage.googleapis.com/uno-y-b48fb.appspot.com/product_images/Risifrutti%20Hallon.png?Expires=2220457345&GoogleAccessId=firebase-adminsdk-rmjcr%40uno-y-b48fb.iam.gserviceaccount.com&Signature=JXZE9m8Iu7nNS4VyaPJbeoq47rnQcpa09LC6pxFxCMlamstlOQvUMZHap%2FBAnElgvmktcukJSTG5LaZOIDkIwopYnbzd5oWV0T%2BUkZIhr%2BF1yWaskI4hcAC17fB4p7ZcdiIB1hARi%2BSVQqdbs3draUZfuDmchvrZqhU%2ByBcjXA4ETcUy8NtdbzVnuYCKKHKt9j8ABfYORuXl1261klT5Sg1iF%2BNzL2PgqjsWDHy1G%2FCtOTv5OtUCIwaWJO5LCfgFWLoJu6hMn3EkMoouLvMbxEkf5Q42Hq4TQfI2Ot1lA1ZgEPjkIJVkSRCzX0O5PDzmwswLMVhbp7F1tZc15puD4w%3D%3D",
  },
]

 