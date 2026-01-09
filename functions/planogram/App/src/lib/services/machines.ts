import { db } from "../firebase.ts";
import { collection, getDocs, getDoc, doc } from "firebase/firestore";
import type { AppMachine, MachineSlot } from "../types.ts";

// Collection name - using backup for now, can switch to app_machines later
const COLLECTION_NAME = "app_machines_backup";

/**
 * Maps w_spec to width field for compatibility with existing components
 */
function mapSlotFields(slot: any): MachineSlot {
  // If w_spec exists, use it as width, otherwise use existing width field
  const width = slot.w_spec ?? slot.width ?? 1;

  return {
    position: slot.position || "",
    product_name: slot.product_name || "",
    image_url: slot.image_url || "",
    price: slot.price || 0,
    stock_current: slot.stock_current || 0,
    stock_max: slot.stock_max || 0,
    category: slot.category || "",
    width: width, // Use w_spec as width
    is_discount: slot.is_discount || false,
    discount: slot.discount || 0,
  };
}

/**
 * Fetches all machines from Firestore
 */
export async function getAllMachines(): Promise<AppMachine[]> {
  try {
    const colRef = collection(db, COLLECTION_NAME);
    const snapshot = await getDocs(colRef);

    if (snapshot.empty) {
      return [];
    }

    const machines = snapshot.docs.map((doc) => {
      const data = doc.data();

      return {
        id: doc.id,
        machine_name: data.machine_name || "",
        machine_model: data.machine_model || "",
        machine_key: data.machine_key || "",
        machine_sub_group: data.machine_sub_group || "",
        last_sale: data.last_sale || "",
        n_sales: data.n_sales || 0,
        refillers: data.refillers || [],
        slots: (data.slots || []).map(mapSlotFields),
        location: data.location,
      };
    });

    return machines;
  } catch (error) {
    console.error("Error fetching machines:", error);
    throw new Error(`Failed to fetch machines: ${error}`);
  }
}

/**
 * Fetches a single machine by ID
 */
export async function getMachineById(machineId: string): Promise<AppMachine | null> {
  if (!machineId) {
    throw new Error("Machine ID is required");
  }

  try {
    const docRef = doc(db, COLLECTION_NAME, machineId);
    const snapshot = await getDoc(docRef);

    if (snapshot.exists()) {
      const data = snapshot.data();

      return {
        id: snapshot.id,
        machine_name: data.machine_name || "",
        machine_model: data.machine_model || "",
        machine_key: data.machine_key || "",
        machine_sub_group: data.machine_sub_group || "",
        last_sale: data.last_sale || "",
        n_sales: data.n_sales || 0,
        refillers: data.refillers || [],
        slots: (data.slots || []).map(mapSlotFields),
        location: data.location,
      };
    } else {
      return null;
    }
  } catch (error) {
    console.error(`Error fetching machine ${machineId}:`, error);
    throw new Error(`Failed to fetch machine: ${error}`);
  }
}
