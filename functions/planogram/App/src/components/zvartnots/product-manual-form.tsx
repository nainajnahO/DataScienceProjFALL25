import { useState } from "react"
import type { Product } from "../../lib/types"

interface ProductManualFormProps {
  onProductCreate: (product: Product) => void
  onCancel: () => void
}

export default function ProductManualForm({ onProductCreate, onCancel: _onCancel }: ProductManualFormProps) {
  const [newProduct, setNewProduct] = useState({ name: "", price: "", category: "", image: "" })

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement("canvas")
        const ctx = canvas.getContext("2d")

        let width = img.width
        let height = img.height
        const MAX_SIZE = 300

        if (width > height) {
          if (width > MAX_SIZE) {
            height *= MAX_SIZE / width
            width = MAX_SIZE
          }
        } else {
          if (height > MAX_SIZE) {
            width *= MAX_SIZE / height
            height = MAX_SIZE
          }
        }

        canvas.width = width
        canvas.height = height

        if (ctx) {
          ctx.drawImage(img, 0, 0, width, height)
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7)
          setNewProduct(prev => ({ ...prev, image: dataUrl }))
        }
      }
      img.src = event.target?.result as string
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = () => {
    const product: Product = {
      id: `custom-${Date.now()}`,
      name: newProduct.name,
      product_name: newProduct.name,
      price: Number(newProduct.price),
      category: newProduct.category || "Custom",
      width: 1,
      image: newProduct.image,
    }
    onProductCreate(product)
  }

  return (
    <div className="flex flex-col gap-4 p-1">
      <div>
        <label className="block text-sm font-medium mb-1">Product Name</label>
        <input
          type="text"
          value={newProduct.name}
          onChange={e => setNewProduct({ ...newProduct, name: e.target.value })}
          className="w-full px-3 py-2 bg-background border border-input rounded-lg"
          placeholder="e.g. Cola Zero"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-1">Price (kr)</label>
        <input
          type="number"
          value={newProduct.price}
          onChange={e => setNewProduct({ ...newProduct, price: e.target.value })}
          className="w-full px-3 py-2 bg-background border border-input rounded-lg"
          placeholder="25"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-1">Category</label>
        <input
          type="text"
          value={newProduct.category}
          onChange={e => setNewProduct({ ...newProduct, category: e.target.value })}
          className="w-full px-3 py-2 bg-background border border-input rounded-lg"
          placeholder="Snacks"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">Product Image</label>
        <div className="flex items-center gap-4">
          {newProduct.image && (
            <img
              src={newProduct.image}
              alt="Preview"
              className="w-16 h-16 object-contain border rounded bg-white"
            />
          )}
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="text-sm text-foreground file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-secondary file:text-secondary-foreground hover:file:bg-secondary/80"
          />
        </div>
      </div>

      <button
        disabled={!newProduct.name || !newProduct.price}
        onClick={handleSubmit}
        className="w-full py-3 bg-primary text-primary-foreground font-bold rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed mt-4"
      >
        Add Product
      </button>
    </div>
  )
}

